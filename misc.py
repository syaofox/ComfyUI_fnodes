from comfy.utils import common_upscale

from .utils.mask_utils import mask_floor, mask_unsqueeze

_CATEGORY = 'fnodes/misc'


class DisplayAny:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'input': (('*', {})),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ('STRING',)
    FUNCTION = 'execute'
    OUTPUT_NODE = True

    CATEGORY = _CATEGORY

    def execute(self, input):
        text = str(input)

        return {'ui': {'text': text}, 'result': (text,)}


class PrimitiveText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'string': ('STRING', {'multiline': True, 'default': ''}),
            }
        }

    CATEGORY = _CATEGORY
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('text',)

    FUNCTION = 'execute'

    def execute(self, string=''):
        return (string,)


class FillMaskedImageArea:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'mask': ('MASK',),
                'fill': ('FLOAT', {'default': 0, 'min': 0, 'max': 1, 'step': 0.01}),
            }
        }

    RETURN_TYPES = ('IMAGE',)
    CATEGORY = _CATEGORY
    FUNCTION = 'fill'
    DESCRIPTION = '填充图像区域'

    def fill(self, image, mask, fill):
        image = image.detach().clone()
        alpha = mask_unsqueeze(mask_floor(mask))
        assert alpha.shape[0] == image.shape[0], 'Image and mask batch size does not match'

        if image.shape[1] != mask.shape[1] or image.shape[2] != mask.shape[2]:
            samples = image.movedim(-1, 1)
            image = common_upscale(samples, mask.shape[2], mask.shape[1], 'lanczos', 'disabled')
            image = image.movedim(1, -1)

        m = (1.0 - alpha).squeeze(1)
        for i in range(3):
            image[:, :, :, i] -= fill
            image[:, :, :, i] *= m
            image[:, :, :, i] += fill

        return (image,)


MISC_CLASS_MAPPINGS = {
    'DisplayAny-': DisplayAny,
    'PrimitiveText-': PrimitiveText,
    'FillMaskedImageArea-': FillMaskedImageArea,
}

MISC_NAME_MAPPINGS = {
    'DisplayAny-': 'Display Any',
    'PrimitiveText-': 'Primitive Text',
    'FillMaskedImageArea-': 'Fill Masked Image Area',
}
