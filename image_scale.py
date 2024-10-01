import math

import cv2

from comfy.utils import common_upscale

from .utils.image_convert import mask2tensor, np2tensor, tensor2mask, tensor2np
from .utils.mask_utils import solid_mask
from .utils.utils import make_even

_CATEGORY = 'fnodes/image scale'
UPSCALE_METHODS = ['lanczos', 'nearest-exact', 'bilinear', 'area', 'bicubic']


class GetImageSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
            }
        }

    RETURN_TYPES = (
        'INT',
        'INT',
        'INT',
    )
    RETURN_NAMES = (
        'width',
        'height',
        'count',
    )
    FUNCTION = 'execute'
    CATEGORY = _CATEGORY

    def execute(self, image):
        return {
            'ui': {
                'width': (image.shape[2],),
                'height': (image.shape[1],),
                'count': (image.shape[0],),
            },
            'result': (
                image.shape[2],
                image.shape[1],
                image.shape[0],
            ),
        }


class BaseImageScaler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'upscale_method': (UPSCALE_METHODS,),
            },
            'optional': {
                'mask': ('MASK',),
            },
        }

    RETURN_TYPES = ('IMAGE', 'MASK', 'INT', 'INT', 'INT')
    RETURN_NAMES = ('image', 'mask', 'width', 'height', 'min_dimension')
    CATEGORY = _CATEGORY

    def scale_image(self, image, width, height, upscale_method, mask=None):
        image_tensor = image.movedim(-1, 1)
        scaled_image = common_upscale(image_tensor, width, height, upscale_method, 'disabled')
        scaled_image = scaled_image.movedim(1, -1)

        result_mask = solid_mask(width, height)
        if mask is not None:
            mask_image = mask2tensor(mask)
            mask_image = mask_image.movedim(-1, 1)
            mask_image = common_upscale(mask_image, width, height, upscale_method, 'disabled')
            mask_image = mask_image.movedim(1, -1)
            result_mask = tensor2mask(mask_image)

        return scaled_image, result_mask

    def prepare_result(self, scaled_image, result_mask, width, height):
        return {
            'ui': {
                'width': (width,),
                'height': (height,),
            },
            'result': (
                scaled_image,
                result_mask,
                width,
                height,
                min(width, height),
            ),
        }


class ImageScalerForSDModels(BaseImageScaler):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs['required']['sd_model_type'] = (['sdxl', 'sd15', 'sd15+', 'sdxl+'],)
        return base_inputs

    FUNCTION = 'execute'
    DESCRIPTION = """
    根据SD模型类型缩放图片到指定像素数，sd15为512x512，sd15+为512x768，sdxl为1024x1024，sdxl+为1024x1280
    """

    def execute(self, image, upscale_method, sd_model_type, mask=None):
        sd_dimensions = {'sd15': (512, 512), 'sd15+': (512, 768), 'sdxl': (1024, 1024), 'sdxl+': (1024, 1280)}
        target_width, target_height = sd_dimensions.get(sd_model_type, (1024, 1024))
        total_pixels = target_width * target_height

        scale_by = math.sqrt(total_pixels / (image.shape[2] * image.shape[1]))
        width = round(image.shape[2] * scale_by)
        height = round(image.shape[1] * scale_by)

        scaled_image, result_mask = self.scale_image(image, width, height, upscale_method, mask)
        return self.prepare_result(scaled_image, result_mask, width, height)


class ImageScaleBySpecifiedSide(BaseImageScaler):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs['required'].update(
            {
                'size': ('INT', {'default': 512, 'min': 0, 'step': 1, 'max': 99999}),
                'shorter': ('BOOLEAN', {'default': False}),
            }
        )
        return base_inputs

    FUNCTION = 'execute'
    DESCRIPTION = """
    根据指定边长缩放图片，shorter为True时参照短边，否则参照长边
    """

    def execute(self, image, size, upscale_method, shorter, mask=None):
        if shorter:
            reference_side_length = min(image.shape[2], image.shape[1])
        else:
            reference_side_length = max(image.shape[2], image.shape[1])

        scale_by = reference_side_length / size
        width = make_even(round(image.shape[2] / scale_by))
        height = make_even(round(image.shape[1] / scale_by))

        scaled_image, result_mask = self.scale_image(image, width, height, upscale_method, mask)
        return self.prepare_result(scaled_image, result_mask, width, height)


class ComputeImageScaleRatio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'target_max_size': (
                    'INT',
                    {'default': 1920, 'min': 0, 'step': 1, 'max': 99999},
                ),
            },
        }

    RETURN_TYPES = (
        'FLOAT',
        'INT',
        'INT',
    )
    RETURN_NAMES = (
        'rescale_ratio',
        'width',
        'height',
    )
    FUNCTION = 'execute'
    CATEGORY = _CATEGORY
    DESCRIPTION = '根据引用图片的大小和目标最大尺寸，返回缩放比例和缩放后的宽高'

    def execute(self, image, target_max_size):
        samples = image.movedim(-1, 1)
        width, height = samples.shape[3], samples.shape[2]

        rescale_ratio = target_max_size / max(width, height)

        new_width = make_even(round(width * rescale_ratio))
        new_height = make_even(round(height * rescale_ratio))

        return {
            'ui': {
                'rescale_ratio': (rescale_ratio,),
                'width': (new_width,),
                'height': (new_height,),
            },
            'result': (
                rescale_ratio,
                new_width,
                new_height,
            ),
        }


class ImageRotate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image_from': ('IMAGE',),
                'angle': (
                    'FLOAT',
                    {'default': 0.1, 'min': -14096, 'max': 14096, 'step': 0.01},
                ),
                'expand': ('BOOLEAN', {'default': True}),
            },
        }

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('rotated_image',)
    FUNCTION = 'run'
    CATEGORY = _CATEGORY

    def run(self, image_from, angle, expand):
        image_np = tensor2np(image_from[0])

        height, width = image_np.shape[:2]
        center = (width / 2, height / 2)

        if expand:
            # 计算新图像的尺寸
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            abs_cos = abs(rot_mat[0, 0])
            abs_sin = abs(rot_mat[0, 1])
            new_width = int(height * abs_sin + width * abs_cos)
            new_height = int(height * abs_cos + width * abs_sin)

            # 调整旋转矩阵
            rot_mat[0, 2] += (new_width / 2) - center[0]
            rot_mat[1, 2] += (new_height / 2) - center[1]

            # 执行旋转
            rotated_image = cv2.warpAffine(image_np, rot_mat, (new_width, new_height), flags=cv2.INTER_CUBIC)
        else:
            # 不扩展图像尺寸的旋转
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(image_np, rot_mat, (width, height), flags=cv2.INTER_CUBIC)

        # 转换回tensor格式
        rotated_tensor = np2tensor(rotated_image).unsqueeze(0)

        return (rotated_tensor,)


IMAGE_SCALE_CLASS_MAPPINGS = {
    'GetImageSize-': GetImageSize,
    'ImageScalerForSDModels-': ImageScalerForSDModels,
    'ImageScaleBySpecifiedSide-': ImageScaleBySpecifiedSide,
    'ComputeImageScaleRatio-': ComputeImageScaleRatio,
    'ImageRotate-': ImageRotate,
}

IMAGE_SCALE_NAME_MAPPINGS = {
    'GetImageSize-': 'Get Image Size',
    'ImageScalerForSDModels-': 'Image Scaler for SD Models',
    'ImageScaleBySpecifiedSide-': 'Image Scale By Specified Side',
    'ComputeImageScaleRatio-': 'Compute Image Scale Ratio',
    'ImageRotate-': 'Image Rotate',
}
