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


MISC_CLASS_MAPPINGS = {
    'DisplayAny-': DisplayAny,
    'PrimitiveText-': PrimitiveText,
}

MISC_NAME_MAPPINGS = {
    'DisplayAny-': 'Display Any',
    'PrimitiveText-': 'Primitive Text',
}
