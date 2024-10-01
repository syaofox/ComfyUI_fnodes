# from custom_nodes.ComfyUI_FaceAnalysis.dlib import dlib

from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import torch
from PIL import Image

import folder_paths
from comfy.utils import ProgressBar

from .utils.downloader import download_model
from .utils.image_convert import np2tensor, pil2mask, pil2tensor, tensor2np, tensor2pil
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

        if fill:
            mask = fill_holes(mask)

        grow_count = int(grow_percent * max(mask.shape)) + grow
        if grow_count > 0:
            mask = expand_mask(mask, grow_count, grow_tapered)

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

    RETURN_TYPES = ('IMAGE', 'MASK', 'MASK', 'FLOAT', 'FLOAT')
    RETURN_NAMES = ('aligned_image', 'mask', 'inverted_mask', 'rotation_angle', 'inverse_rotation_angle')
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

        original_image = tensor2pil(image_from[0]) if not is_flipped else Image.fromarray(processed_image)

        # 创建并旋转遮罩
        mask = Image.new('L', original_image.size, 255)
        rotated_mask = mask.rotate(rotation_angle, expand=expand, resample=Image.Resampling.BICUBIC)
        mask_tensor = pil2mask(rotated_mask)

        # 旋转原始图像
        aligned_image = original_image.rotate(rotation_angle, expand=expand, resample=Image.Resampling.BICUBIC)
        aligned_image_tensor = pil2tensor(aligned_image)

        if is_flipped:
            rotation_angle += 180

        return (aligned_image_tensor, mask_tensor, 1.0 - mask_tensor, rotation_angle, 360 - rotation_angle)


FACE_ANALYSIS_CLASS_MAPPINGS = {
    'GeneratePreciseFaceMask-': GeneratePreciseFaceMask,
    'AlignImageByFace-': AlignImageByFace,
}

FACE_ANALYSIS_NAME_MAPPINGS = {
    'GeneratePreciseFaceMask-': 'Generate PreciseFaceMask',
    'AlignImageByFace-': 'Align Image By Face',
}
