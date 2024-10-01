import math

from comfy.utils import common_upscale

from .utils.image_convert import mask2tensor, tensor2mask
from .utils.mask_utils import solid_mask

_CATEGORY = 'fnodes/images'


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


class ImageScalerForSDModels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'upscale_method': (['lanczos', 'nearest-exact', 'bilinear', 'area', 'bicubic'],),
                'sd_model_type': (['sdxl', 'sd15', 'sd15+', 'sdxl+'],),
            },
            'optional': {
                'mask': ('MASK',),
            },
        }

    RETURN_TYPES = (
        'IMAGE',
        'MASK',
        'INT',
        'INT',
        'INT',
    )
    RETURN_NAMES = ('image', 'mask', 'width', 'height', 'min_dimension')
    FUNCTION = 'execute'
    CATEGORY = _CATEGORY
    DESCRIPTION = """
    根据SD模型类型缩放图片到指定像素数，sd15为512x512，sd15+为512x768，sdxl为1024x1024，sdxl+为1024x1280
    """

    def execute(self, image, upscale_method, sd_model_type, mask=None):
        image_tensor = image.movedim(-1, 1)

        sd_dimensions = {'sd15': (512, 512), 'sd15+': (512, 768), 'sdxl': (1024, 1024), 'sdxl+': (1024, 1280)}

        target_width, target_height = sd_dimensions.get(sd_model_type, (1024, 1024))
        total_pixels = target_width * target_height

        scale_by = math.sqrt(total_pixels / (image_tensor.shape[3] * image_tensor.shape[2]))
        width = round(image_tensor.shape[3] * scale_by)
        height = round(image_tensor.shape[2] * scale_by)

        scaled_image = common_upscale(image_tensor, width, height, upscale_method, 'disabled')
        scaled_image = scaled_image.movedim(1, -1)

        result_mask = solid_mask(width, height)

        if mask is not None:
            mask_image = mask2tensor(mask)
            mask_image = mask_image.movedim(-1, 1)
            mask_image = common_upscale(mask_image, width, height, upscale_method, 'disabled')
            mask_image = mask_image.movedim(1, -1)
            result_mask = tensor2mask(mask_image)

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


IMAGE_CLASS_MAPPINGS = {
    'ImageScalerForSDModels-': ImageScalerForSDModels,
    'GetImageSize-': GetImageSize,
}

IMAGE_NAME_MAPPINGS = {
    'ImageScalerForSDModels-': 'Image Scaler for SD Models',
    'GetImageSize-': 'Get Image Size',
}
