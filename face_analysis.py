import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import torch
from PIL import Image, ImageDraw, ImageFilter

import folder_paths
from comfy.utils import ProgressBar, common_upscale

from .Face_morph import FaceMorph
from .utils.downloader import download_model
from .utils.image_convert import np2tensor, pil2mask, pil2tensor, tensor2mask, tensor2np, tensor2pil
from .utils.mask_utils import blur_mask, expand_mask, fill_holes, invert_mask

_CATEGORY = 'fnodes/face_analysis'


class Occluder:
    def __init__(self, occluder_model_path):
        self.occluder_model_path = occluder_model_path
        self.face_occluder = self.get_face_occluder()

    def get_face_occluder(self):
        return onnxruntime.InferenceSession(
            self.occluder_model_path,
            providers=['CPUExecutionProvider'],
        )

    def create_occlusion_mask(self, crop_vision_frame):
        prepare_vision_frame = cv2.resize(crop_vision_frame, self.face_occluder.get_inputs()[0].shape[1:3][::-1])
        prepare_vision_frame = np.expand_dims(prepare_vision_frame, axis=0).astype(np.float32) / 255
        prepare_vision_frame = prepare_vision_frame.transpose(0, 1, 2, 3)
        occlusion_mask = self.face_occluder.run(None, {self.face_occluder.get_inputs()[0].name: prepare_vision_frame})[0][0]
        occlusion_mask = occlusion_mask.transpose(0, 1, 2).clip(0, 1).astype(np.float32)
        occlusion_mask = cv2.resize(occlusion_mask, crop_vision_frame.shape[:2][::-1])
        occlusion_mask = (cv2.GaussianBlur(occlusion_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
        return occlusion_mask


class GeneratePreciseFaceMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'input_image': ('IMAGE',),
            },
            'optional': {
                'grow': ('INT', {'default': 0, 'min': -4096, 'max': 4096, 'step': 1}),
                'grow_percent': (
                    'FLOAT',
                    {'default': 0.00, 'min': 0.00, 'max': 2.0, 'step': 0.01},
                ),
                'grow_tapered': ('BOOLEAN', {'default': False}),
                'blur': ('INT', {'default': 0, 'min': 0, 'max': 4096, 'step': 1}),
                'fill': ('BOOLEAN', {'default': False}),
            },
        }

    RETURN_TYPES = (
        'MASK',
        'MASK',
        'IMAGE',
    )
    RETURN_NAMES = (
        'mask',
        'inverted_mask',
        'image',
    )
    FUNCTION = 'generate_mask'
    CATEGORY = _CATEGORY
    DESCRIPTION = '生成精确人脸遮罩'

    def _load_occluder_model(self):
        """加载人脸遮挡模型"""
        model_url = 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/dfl_xseg.onnx'
        save_loc = Path(folder_paths.models_dir) / 'fnodes' / 'occluder'
        model_name = 'occluder.onnx'
        download_model(model_url, save_loc, model_name)
        return Occluder(str(save_loc / model_name))

    def generate_mask(self, input_image, grow, grow_percent, grow_tapered, blur, fill):
        face_occluder_model = self._load_occluder_model()

        out_mask, out_inverted_mask, out_image = [], [], []

        steps = input_image.shape[0]
        if steps > 1:
            pbar = ProgressBar(steps)

        for i in range(steps):
            mask, processed_img = self._process_single_image(input_image[i], face_occluder_model, grow, grow_percent, grow_tapered, blur, fill)
            out_mask.append(mask)
            out_inverted_mask.append(invert_mask(mask))
            out_image.append(processed_img)
            if steps > 1:
                pbar.update(1)

        return torch.stack(out_mask).squeeze(-1), torch.stack(out_inverted_mask).squeeze(-1), torch.stack(out_image)

    def _process_single_image(self, img, face_occluder_model, grow, grow_percent, grow_tapered, blur, fill):
        """处理单张图像"""
        face = tensor2np(img)
        if face is None:
            print('\033[96mNo face detected\033[0m')
            return torch.zeros_like(img)[:, :, :1], torch.zeros_like(img)

        cv2_image = cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR)
        occlusion_mask = face_occluder_model.create_occlusion_mask(cv2_image)

        if occlusion_mask is None:
            print('\033[96mNo landmarks detected\033[0m')
            return torch.zeros_like(img)[:, :, :1], torch.zeros_like(img)

        mask = self._process_mask(occlusion_mask, img, grow, grow_percent, grow_tapered, blur, fill)
        processed_img = img * mask.repeat(1, 1, 3)
        return mask, processed_img

    def _process_mask(self, occlusion_mask, img, grow, grow_percent, grow_tapered, blur, fill):
        """处理遮罩"""
        mask = np2tensor(occlusion_mask).unsqueeze(0).squeeze(-1).clamp(0, 1).to(device=img.device)

        grow_count = int(grow_percent * max(mask.shape)) + grow
        if grow_count > 0:
            mask = expand_mask(mask, grow_count, grow_tapered)

        if fill:
            mask = fill_holes(mask)

        if blur > 0:
            mask = blur_mask(mask, blur)

        return mask.squeeze(0).unsqueeze(-1)


class AlignImageByFace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'analysis_models': ('ANALYSIS_MODELS',),
                'image_from': ('IMAGE',),
                'expand': ('BOOLEAN', {'default': True}),
                'simple_angle': ('BOOLEAN', {'default': False}),
            },
            'optional': {
                'image_to': ('IMAGE',),
            },
        }

    RETURN_TYPES = ('IMAGE', 'FLOAT', 'FLOAT')
    RETURN_NAMES = ('aligned_image', 'rotation_angle', 'inverse_rotation_angle')
    FUNCTION = 'align'
    CATEGORY = _CATEGORY
    DESCRIPTION = '根据图像中的人脸进行旋转对齐'

    def align(self, analysis_models, image_from, expand=True, simple_angle=False, image_to=None):
        source_image = tensor2np(image_from[0])

        def find_nearest_angle(angle):
            angles = [-360, -270, -180, -90, 0, 90, 180, 270, 360]
            normalized_angle = angle % 360
            return min(angles, key=lambda x: min(abs(x - normalized_angle), abs(x - normalized_angle - 360), abs(x - normalized_angle + 360)))

        def calculate_angle(shape):
            left_eye, right_eye = shape[:2]
            return float(np.degrees(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])))

        def detect_face(img, flip=False):
            if flip:
                img = Image.fromarray(img).rotate(180, expand=expand, resample=Image.Resampling.BICUBIC)
                img = np.array(img)
            face_shape = analysis_models.get_keypoints(img)
            return face_shape, img

        # 尝试检测人脸，如果失败则翻转图像再次尝试
        face_shape, processed_image = detect_face(source_image)
        if face_shape is None:
            face_shape, processed_image = detect_face(source_image, flip=True)
            is_flipped = True
            if face_shape is None:
                raise Exception('无法在图像中检测到人脸。')
        else:
            is_flipped = False

        rotation_angle = calculate_angle(face_shape)
        if simple_angle:
            rotation_angle = find_nearest_angle(rotation_angle)

        # 如果提供了目标图像，计算相对旋转角度
        if image_to is not None:
            target_shape = analysis_models.get_keypoints(tensor2np(image_to[0]))
            if target_shape is not None:
                print(f'目标图像人脸关键点: {target_shape}')
                rotation_angle -= calculate_angle(target_shape)

        original_image = tensor2np(image_from[0]) if not is_flipped else processed_image

        rows, cols = original_image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)

        if expand:
            # 计算新的边界以确保整个图像都包含在内
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_cols = int((rows * sin) + (cols * cos))
            new_rows = int((rows * cos) + (cols * sin))
            M[0, 2] += (new_cols / 2) - cols / 2
            M[1, 2] += (new_rows / 2) - rows / 2
            new_size = (new_cols, new_rows)
        else:
            new_size = (cols, rows)

        aligned_image = cv2.warpAffine(original_image, M, new_size, flags=cv2.INTER_LINEAR)

        # 转换为张量

        aligned_image_tensor = np2tensor(aligned_image).unsqueeze(0)

        if is_flipped:
            rotation_angle += 180

        return (aligned_image_tensor, rotation_angle, 360 - rotation_angle)


class FaceCutout:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'analysis_models': ('ANALYSIS_MODELS',),
                'image': ('IMAGE',),
                'padding': ('INT', {'default': 0, 'min': 0, 'max': 4096, 'step': 1}),
                'padding_percent': ('FLOAT', {'default': 0.1, 'min': 0.0, 'max': 2.0, 'step': 0.01}),
                'face_index': ('INT', {'default': -1, 'min': -1, 'max': 4096, 'step': 1}),
                'rescale_mode': (['sdxl', 'sd15', 'sdxl+', 'sd15+', 'none', 'custom'],),
                'custom_megapixels': ('FLOAT', {'default': 1.0, 'min': 0.01, 'max': 16.0, 'step': 0.01}),
            },
        }

    RETURN_TYPES = ('IMAGE', 'BOUNDINGINFO')
    RETURN_NAMES = ('cutout_image', 'bounding_info')
    FUNCTION = 'execute'
    CATEGORY = _CATEGORY
    DESCRIPTION = '切下人脸并进行缩放'

    def execute(self, analysis_models, image, padding, padding_percent, rescale_mode, custom_megapixels, face_index=-1):
        target_size = self._get_target_size(rescale_mode, custom_megapixels)

        img = image[0]

        pil_image = tensor2pil(img)

        faces, x_coords, y_coords, widths, heights = analysis_models.get_bbox(pil_image, padding, padding_percent)

        face_count = len(faces)
        if face_count == 0:
            raise Exception('未在图像中检测到人脸。')

        if face_index == -1:
            face_index = 0

        face_index = min(face_index, face_count - 1)

        face = faces[face_index]
        x = x_coords[face_index]
        y = y_coords[face_index]
        w = widths[face_index]
        h = heights[face_index]

        scale_factor = 1

        if target_size > 0:
            scale_factor = math.sqrt(target_size / (w * h))
            new_width = round(w * scale_factor)
            new_height = round(h * scale_factor)
            face = self._rescale_image(face, new_width, new_height)

        bounding_info = {
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'scale_factor': scale_factor,
        }

        return (face, bounding_info)

    @staticmethod
    def _get_target_size(rescale_mode, custom_megapixels):
        if rescale_mode == 'custom':
            return int(custom_megapixels * 1024 * 1024)
        size_map = {'sd15': 512 * 512, 'sd15+': 512 * 768, 'sdxl': 1024 * 1024, 'sdxl+': 1024 * 1280, 'none': -1}
        return size_map.get(rescale_mode, -1)

    @staticmethod
    def _rescale_image(image, width, height):
        samples = image.movedim(-1, 1)
        resized = common_upscale(samples, width, height, 'lanczos', 'disabled')
        return resized.movedim(1, -1)


class FacePaste:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'destination': ('IMAGE',),
                'source': ('IMAGE',),
                'bounding_info': ('BOUNDINGINFO',),
                'margin': ('INT', {'default': 0, 'min': 0, 'max': 4096, 'step': 1}),
                'margin_percent': ('FLOAT', {'default': 0.10, 'min': 0.0, 'max': 2.0, 'step': 0.05}),
                'blur_radius': ('INT', {'default': 10, 'min': 0, 'max': 4096, 'step': 1}),
            },
        }

    RETURN_TYPES = ('IMAGE', 'MASK')
    RETURN_NAMES = ('image', 'mask')
    FUNCTION = 'paste'
    CATEGORY = _CATEGORY
    DESCRIPTION = '将人脸图像贴回原图'

    @staticmethod
    def create_soft_edge_mask(size, margin, blur_radius):
        mask = Image.new('L', size, 255)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(((0, 0), size), outline='black', width=margin)
        return mask.filter(ImageFilter.GaussianBlur(blur_radius))

    def paste(self, destination, source, bounding_info, margin, margin_percent, blur_radius):
        if not bounding_info:
            return destination, None

        destination = tensor2pil(destination[0])
        source = tensor2pil(source[0])

        if bounding_info.get('scale_factor', 1) != 1:
            new_size = (bounding_info['width'], bounding_info['height'])
            source = source.resize(new_size, resample=Image.Resampling.LANCZOS)

        ref_size = max(source.width, source.height)
        margin_border = int(ref_size * margin_percent) + margin

        mask = self.create_soft_edge_mask(source.size, margin_border, blur_radius)

        position = (bounding_info['x'], bounding_info['y'])
        destination.paste(source, position, mask)

        return pil2tensor(destination), pil2mask(mask)


class ExtractBoundingBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'bounding_info': ('BOUNDINGINFO',),
            },
        }

    RETURN_TYPES = ('INT', 'INT', 'INT', 'INT')
    RETURN_NAMES = ('x', 'y', 'width', 'height')
    FUNCTION = 'extract'
    CATEGORY = _CATEGORY
    DESCRIPTION = '从边界框信息中提取坐标和尺寸'

    def extract(self, bounding_info):
        return (bounding_info['x'], bounding_info['y'], bounding_info['width'], bounding_info['height'])


# 更新类映射
FACE_ANALYSIS_CLASS_MAPPINGS = {
    'GeneratePreciseFaceMask-': GeneratePreciseFaceMask,
    'AlignImageByFace-': AlignImageByFace,
    'FaceCutout-': FaceCutout,
    'FacePaste-': FacePaste,
    'ExtractBoundingBox-': ExtractBoundingBox,
    'FaceMorph-': FaceMorph,
}

FACE_ANALYSIS_NAME_MAPPINGS = {
    'GeneratePreciseFaceMask-': 'Generate PreciseFaceMask',
    'AlignImageByFace-': 'Align Image By Face',
    'FaceCutout-': 'Face Cutout',
    'FacePaste-': 'Face Paste',
    'ExtractBoundingBox-': 'Extract Bounding Box',
    'FaceMorph-': 'Face Morph',
}
