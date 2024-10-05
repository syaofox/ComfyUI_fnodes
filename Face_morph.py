from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator

import folder_paths

from .liveportrait.utils.cropper import CropperMediaPipe
from .utils.downloader import download_model
from .utils.image_convert import pil2tensor

_CATEGORY = 'fnodes/face_analysis'


class FaceLandmarkExtractor(CropperMediaPipe):
    def extract_face_landmarks(self, img_rgb, face_index):
        landmark_info = {}
        face_result = self.lmk_extractor(img_rgb)
        if face_result is None:
            raise Exception('未在图像中检测到人脸。')
        face_landmarks = face_result[face_index]
        lmks = [[face_landmarks[index].x * img_rgb.shape[1], face_landmarks[index].y * img_rgb.shape[0]] for index in range(len(face_landmarks))]
        recon_ret = self.landmark_runner.run(img_rgb, np.array(lmks))
        landmark_info['landmarks'] = recon_ret['pts']
        return landmark_info


class FaceMorph:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'source_image': ('IMAGE',),
                'target_image': ('IMAGE',),
                'landmark_type': (['ALL', 'OUTLINE'],),
                'align_type': (['Width', 'Height', 'Landmarks', 'JawLine'],),
                'onnx_device': (['CPU', 'CUDA', 'ROCM', 'CoreML', 'torch_gpu'], {'default': 'CPU'}),
            },
        }

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('warped_image',)
    FUNCTION = 'execute'
    CATEGORY = _CATEGORY
    DESCRIPTION = '根据输入的源图像和目标图像，进行人脸液化变形'

    def __init__(self):
        self.landmark_extractor = None
        self.current_config = None

    def landmark203_to_68(self, source):
        out = []
        jaw_indices = [108, 126, 144]
        for start in jaw_indices:
            out.append(source[start])
            out.extend(
                [
                    (source[start + i] * 3 + source[start + i + 1]) / 4 if i % 4 == 2 else (source[start + i] + source[start + i + 1]) / 2 if i % 4 == 0 else (source[start + i] + source[start + i + 1] * 3) / 4 if i % 4 == 2 else source[start + i]
                    for i in range(2, 17, 2)
                ]
            )

        eyebrow_indices = [(145, 162), (165, 182)]
        for start, end in eyebrow_indices:
            out.append(source[start])
            out.extend([(source[start + i] + source[end - i]) / 2 for i in range(3, 10, 3)])
            out.append(source[start + 10])

        nose_indices = [199, 200, 201, 189, 190, 202, 191, 192]
        out.append(source[199])
        out.append((source[199] + source[200]) / 2)
        out.extend([source[i] for i in nose_indices[1:]])

        eye_indices = [(0, 21), (24, 45)]
        for start, end in eye_indices:
            out.extend([source[i] for i in range(start, end, 4)])

        lip_indices = [48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105]
        out.extend([source[i] for i in lip_indices])

        return out

    def initialize_landmark_extractor(self, onnx_device):
        extractor_init_config = {'keep_model_loaded': True, 'onnx_device': onnx_device}
        if self.landmark_extractor is None or self.current_config != extractor_init_config:
            self.current_config = extractor_init_config
            self.landmark_extractor = FaceLandmarkExtractor(**extractor_init_config)

    def process_image(self, image):
        image_np = (image.contiguous() * 255).byte().numpy()
        if self.landmark_extractor is None:
            raise ValueError('self.landmark_extractor 未初始化')
        landmark_info = self.landmark_extractor.extract_face_landmarks(image_np[0], 0)
        landmarks = self.landmark203_to_68(landmark_info['landmarks'])
        return image_np[0], np.array(landmarks[:65])

    def calculate_facial_features(self, landmarks):
        return {'left_eye': np.mean(landmarks[36:42], axis=0), 'right_eye': np.mean(landmarks[42:48], axis=0), 'jaw': landmarks[0:17], 'center_of_jaw': np.mean(landmarks[0:17], axis=0)}

    def create_grid_points(self, width, height):
        x = np.linspace(0, width, 16)
        y = np.linspace(0, height, 16)
        xx, yy = np.meshgrid(x, y)
        src_points = np.column_stack((xx.ravel(), yy.ravel()))
        mask = (src_points[:, 0] <= width / 8) | (src_points[:, 0] >= 7 * width / 8) | (src_points[:, 1] >= 7 * height / 8) | (src_points[:, 1] <= height / 8)
        return src_points[mask]

    def calculate_ratios(self, landmarks):
        min_x, max_x = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
        min_y, max_y = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
        ratio = (max_x - min_x) / (max_y - min_y)
        middle_point = [(max_x + min_x) / 2, (max_y + min_y) / 2]
        return ratio, middle_point

    def align_width_height(self, landmarks1, landmarks2, jaw1, jaw2, src_points, dst_points, features1, features2, landmark_type, align_type):
        target_points = landmarks1.copy() if landmark_type == 'ALL' else jaw1.copy()
        dst_points = np.append(dst_points, landmarks1 if landmark_type == 'ALL' else jaw1, axis=0)

        ratio1, middle_point = self.calculate_ratios(landmarks1)
        ratio2, _ = self.calculate_ratios(landmarks2)

        if align_type == 'Width':
            target_points[:, 1] = (target_points[:, 1] - middle_point[1]) * ratio1 / ratio2 + middle_point[1]
        else:  # Height
            target_points[:, 0] = (target_points[:, 0] - middle_point[0]) * ratio2 / ratio1 + middle_point[0]

        return np.append(src_points, target_points, axis=0), dst_points

    def align_landmarks(self, landmarks1, landmarks2, jaw1, jaw2, src_points, dst_points, features1, features2, landmark_type, _):
        if landmark_type == 'ALL':
            middle_of_eyes1 = (features1['left_eye'] + features1['right_eye']) / 2
            middle_of_eyes2 = (features2['left_eye'] + features2['right_eye']) / 2
            factor = np.linalg.norm(features1['left_eye'] - features1['right_eye']) / np.linalg.norm(features2['left_eye'] - features2['right_eye'])
            target_points = (landmarks2 - middle_of_eyes2) * factor + middle_of_eyes1
            target_points[0:17] = (landmarks2[0:17] - features2['center_of_jaw']) * factor + features1['center_of_jaw']
            dst_points = np.append(dst_points, landmarks1, axis=0)
        else:
            target_points = (jaw2 - features2['center_of_jaw']) + features1['center_of_jaw']
            dst_points = np.append(dst_points, jaw1, axis=0)

        return np.append(src_points, target_points, axis=0), dst_points

    def align_jaw_line(self, landmarks1, landmarks2, jaw1, jaw2, src_points, dst_points, features1, features2, landmark_type, _):
        factor = np.linalg.norm(jaw1[0] - jaw1[-1]) / np.linalg.norm(jaw2[0] - jaw2[-1])
        if landmark_type == 'ALL':
            target_points = (landmarks2 - jaw2[0]) * factor + jaw1[0]
            dst_points = np.append(dst_points, landmarks1, axis=0)
        else:
            target_points = (jaw2 - jaw2[0]) * factor + jaw1[0]
            dst_points = np.append(dst_points, jaw1, axis=0)
        return np.append(src_points, target_points, axis=0), dst_points

    def warp_image(self, image, src_points, dst_points):
        height, width = image.shape[:2]
        src_points[:, [0, 1]] = src_points[:, [1, 0]]
        dst_points[:, [0, 1]] = dst_points[:, [1, 0]]

        interp = LinearNDInterpolator(src_points, dst_points)

        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        map_coords = interp(np.column_stack((yy.ravel(), xx.ravel()))).reshape(height, width, 2).astype(np.float32)

        return cv2.remap(image, map_coords[:, :, 1], map_coords[:, :, 0], cv2.INTER_LINEAR)

    def download_models(self):
        save_loc = Path(folder_paths.models_dir) / 'liveportrait'

        models = [
            ('landmark.onnx', 'https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/landmark.onnx?download=true'),
            ('landmark_model.pth', 'https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/landmark_model.pth?download=true'),
        ]

        for model_name, model_url in models:
            download_model(model_url, save_loc, model_name)

    def execute(self, source_image, target_image, landmark_type, align_type, onnx_device):
        self.download_models()

        self.initialize_landmark_extractor(onnx_device)
        image1, landmarks1 = self.process_image(source_image)
        _, landmarks2 = self.process_image(target_image)

        features1 = self.calculate_facial_features(landmarks1)
        features2 = self.calculate_facial_features(landmarks2)

        src_points = self.create_grid_points(*image1.shape[:2][::-1])
        dst_points = src_points.copy()

        align_funcs = {'Width': self.align_width_height, 'Height': self.align_width_height, 'Landmarks': self.align_landmarks, 'JawLine': self.align_jaw_line}

        align_func = align_funcs.get(align_type)
        if align_func:
            src_points, dst_points = align_func(landmarks1, landmarks2, features1['jaw'], features2['jaw'], src_points, dst_points, features1, features2, landmark_type, align_type)

        warped_image = self.warp_image(image1, src_points, dst_points)
        return (pil2tensor(warped_image),)
