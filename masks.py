import torch
from PIL import Image, ImageFilter, ImageOps

from comfy.utils import common_upscale

from .utils.image_convert import mask2tensor, np2tensor, tensor2mask
from .utils.mask_utils import blur_mask, combine_mask, expand_mask, fill_holes, grow_mask, invert_mask

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


class MaskChange:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'mask': ('MASK',),
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

    RETURN_TYPES = ('MASK', 'MASK')
    RETURN_NAMES = ('mask', 'inverted_mask')
    FUNCTION = 'execute'
    CATEGORY = _CATEGORY
    DESCRIPTION = '修改和处理遮罩'

    def execute(self, mask, grow, grow_percent, grow_tapered, blur, fill):
        grow_count = int(grow_percent * max(mask.shape)) + grow
        if grow_count > 0:
            mask = expand_mask(mask, grow_count, grow_tapered)

        if fill:
            mask = fill_holes(mask)

        if blur > 0:
            mask = blur_mask(mask, blur)

        # mask = mask.squeeze(0).unsqueeze(-1)

        return (mask, invert_mask(mask))


class Depth2Mask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image_depth': ('IMAGE',),
                'depth': (
                    'FLOAT',
                    {'default': 0.2, 'min': 0.0, 'max': 1.0, 'step': 0.01, 'round': 0.001, 'display': 'number'},
                ),
            },
        }

    RETURN_TYPES = ('MASK', 'MASK')
    RETURN_NAMES = ('mask', 'mask_inverted')

    FUNCTION = 'execute'

    CATEGORY = _CATEGORY
    DESCRIPTION = '将深度图像转换为遮罩'

    def execute(self, image_depth, depth):
        def upscale(image, upscale_method, width, height):
            samples = image.movedim(-1, 1)
            s = common_upscale(samples, width, height, upscale_method, 'disabled')
            s = s.movedim(1, -1)
            return (s,)

        bs, height, width = image_depth.size()[0], image_depth.size()[1], image_depth.size()[2]

        mask1 = torch.zeros((bs, height, width))

        image_depth = upscale(image_depth, 'lanczos', width, height)[0]

        mask1 = (image_depth[..., 0] < depth).float()

        return mask1, 1.0 - mask1


class MaskScaleBy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'mask': ('MASK',),
                'scale_by': ('FLOAT', {'default': 1.0, 'min': 0.01, 'max': 8.0, 'step': 0.01}),
            }
        }

    RETURN_TYPES = ('MASK',)
    FUNCTION = 'upscale'

    CATEGORY = _CATEGORY

    def upscale(self, mask, scale_by):
        image = mask2tensor(mask)
        samples = image.movedim(-1, 1)
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)
        s = common_upscale(samples, width, height, 'lanczos', 'disabled')
        s = s.movedim(1, -1)
        return (tensor2mask(s),)


class MaskScale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'mask': ('MASK',),
                'width': ('INT', {'default': 512, 'min': 0, 'max': 16384, 'step': 1}),
                'height': ('INT', {'default': 512, 'min': 0, 'max': 16384, 'step': 1}),
            }
        }

    RETURN_TYPES = ('MASK',)
    FUNCTION = 'upscale'

    CATEGORY = _CATEGORY

    def upscale(self, mask, width, height):
        image = mask2tensor(mask)
        if width == 0 and height == 0:
            s = image
        else:
            samples = image.movedim(-1, 1)

            if width == 0:
                width = max(1, round(samples.shape[3] * height / samples.shape[2]))
            elif height == 0:
                height = max(1, round(samples.shape[2] * width / samples.shape[3]))

            s = common_upscale(samples, width, height, 'lanczos', 'disabled')
            s = s.movedim(1, -1)
        return (tensor2mask(s),)


MASK_CLASS_MAPPINGS = {
    'OutlineMask-': OutlineMask,
    'CreateBlurredEdgeMask-': CreateBlurredEdgeMask,
    'MaskChange-': MaskChange,
    'Depth2Mask-': Depth2Mask,
    'MaskScaleBy-': MaskScaleBy,
    'MaskScale-': MaskScale,
}

MASK_NAME_MAPPINGS = {
    'OutlineMask-': 'Outline Mask',
    'CreateBlurredEdgeMask-': 'Create Blurred Edge Mask',
    'MaskChange-': 'Mask Change',
    'Depth2Mask-': 'Depth to Mask',
    'MaskScaleBy-': 'Mask Scale By',
    'MaskScale-': 'Mask Scale',
}
