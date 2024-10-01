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


MASK_CLASS_MAPPINGS = {
    'OutlineMask': OutlineMask,
}

MASK_NAME_MAPPINGS = {
    'OutlineMask': 'Outline Mask',
}
