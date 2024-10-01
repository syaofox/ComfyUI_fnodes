from PIL import Image, ImageFilter, ImageOps

from .utils.image_convert import np2tensor, tensor2mask
from .utils.mask_utils import combine_mask, grow_mask

_CATEGORY = 'fnodes/masks'


class OutlineMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'mask': ('MASK',),
                'outline_width': (
                    'INT',
                    {'default': 10, 'min': 1, 'max': 16384, 'step': 1},
                ),
                'tapered_corners': ('BOOLEAN', {'default': True}),
            }
        }

    RETURN_TYPES = ('MASK',)

    FUNCTION = 'execute'

    CATEGORY = _CATEGORY
    DESCRIPTION = '给遮罩添加轮廓线'

    def execute(self, mask, outline_width, tapered_corners):
        m1 = grow_mask(mask, outline_width, tapered_corners)
        m2 = grow_mask(mask, -outline_width, tapered_corners)

        m3 = combine_mask(m1, m2, 0, 0)

        return (m3,)


class CreateBlurredEdgeMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'width': ('INT', {'default': 1024, 'min': 0, 'max': 14096, 'step': 1}),
                'height': ('INT', {'default': 1024, 'min': 0, 'max': 14096, 'step': 1}),
                'border': ('INT', {'default': 0, 'min': 0, 'max': 4096, 'step': 1}),
                'border_percent': (
                    'FLOAT',
                    {'default': 0.05, 'min': 0.0, 'max': 2.0, 'step': 0.01},
                ),
                'blur_radius': (
                    'INT',
                    {'default': 10, 'min': 0, 'max': 4096, 'step': 1},
                ),
                'blur_radius_percent': (
                    'FLOAT',
                    {'default': 0.00, 'min': 0.0, 'max': 2.0, 'step': 0.01},
                ),
            },
            'optional': {
                'image': ('IMAGE', {'tooltips': '如果未提供图像，将使用输入的宽度和高度创建一个白色图像。'}),
            },
        }

    RETURN_TYPES = ('MASK',)
    FUNCTION = 'execute'
    CATEGORY = _CATEGORY
    DESCRIPTION = '根据指定图片创建模糊遮罩'

    def execute(self, width, height, border, border_percent, blur_radius, blur_radius_percent, image=None):
        if image is not None:
            _, height, width, _ = image.shape

        # 计算边框宽度
        border_width = int(min(width, height) * border_percent + border)

        # 计算内部图像的尺寸
        inner_width = width - 2 * border_width
        inner_height = height - 2 * border_width

        # 创建内部白色图像
        inner_image = Image.new('RGB', (inner_width, inner_height), 'white')

        # 扩展图像，添加黑色边框
        image_with_border = ImageOps.expand(inner_image, border=border_width, fill='black')

        # 计算模糊半径
        blur_radius = int(min(width, height) * blur_radius_percent + blur_radius)

        # 应用高斯模糊
        blurred_image = image_with_border.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # 转换为张量
        blurred_tensor = np2tensor(blurred_image)
        blurred_image = blurred_tensor.unsqueeze(0)

        return (tensor2mask(blurred_image),)


MASK_CLASS_MAPPINGS = {
    'OutlineMask-': OutlineMask,
    'CreateBlurredEdgeMask-': CreateBlurredEdgeMask,
}

MASK_NAME_MAPPINGS = {
    'OutlineMask-': 'Outline Mask',
    'CreateBlurredEdgeMask-': 'Create Blurred Edge Mask',
}
