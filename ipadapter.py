import math

import torch
from PIL import Image

from custom_nodes.ComfyUI_IPAdapter_plus.IPAdapterPlus import (
    WEIGHT_TYPES,
    IPAdapterAdvanced,
    ipadapter_execute,
)
from custom_nodes.ComfyUI_IPAdapter_plus.utils import contrast_adaptive_sharpening

try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T

_CATEGORY = 'fnodes/ipadapter'


class IPAdapterMSLayerWeights:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model_type': (['SDXL', 'SD15'],),
                'L0': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 10, 'step': 0.01}),
                'L1': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 10, 'step': 0.01}),
                'L2': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 10, 'step': 0.01}),
                'L3_Composition': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 10, 'step': 0.01}),
                'L4': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 10, 'step': 0.01}),
                'L5': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 10, 'step': 0.01}),
                'L6_Style': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 10, 'step': 0.01}),
                'L7': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 10, 'step': 0.01}),
                'L8': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 10, 'step': 0.01}),
                'L9': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 10, 'step': 0.01}),
                'L10': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 10, 'step': 0.01}),
                'L11': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 10, 'step': 0.01}),
                'L12': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 10, 'step': 0.01}),
                'L13': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 10, 'step': 0.01}),
                'L14': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 10, 'step': 0.01}),
                'L15': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 10, 'step': 0.01}),
            }
        }

    INPUT_NAME = 'layer_weights'
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('layer_weights',)
    FUNCTION = 'execute'
    CATEGORY = _CATEGORY
    DESCRIPTION = 'IPAdapter Mad Scientist Layer Weights'

    def execute(self, model_type, L0, L1, L2, L3_Composition, L4, L5, L6_Style, L7, L8, L9, L10, L11, L12, L13, L14, L15):
        if model_type == 'SD15':
            return (f'0:{L0}, 1:{L1}, 2:{L2}, 3:{L3_Composition}, 4:{L4}, 5:{L5}, 6:{L6_Style}, 7:{L7}, 8:{L8}, 9:{L9}, 10:{L10}, 11:{L11},12:{L12},13:{L13},14:{L14},15:{L15}',)
        else:
            return (f'0:{L0}, 1:{L1}, 2:{L2}, 3:{L3_Composition}, 4:{L4}, 5:{L5}, 6:{L6_Style}, 7:{L7}, 8:{L8}, 9:{L9}, 10:{L10}, 11:{L11}',)


class IPAdapterMSTiled(IPAdapterAdvanced):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL',),
                'ipadapter': ('IPADAPTER',),
                'image': ('IMAGE',),
                'weight': ('FLOAT', {'default': 1.0, 'min': -1, 'max': 5, 'step': 0.05}),
                'weight_faceidv2': ('FLOAT', {'default': 1.0, 'min': -1, 'max': 5.0, 'step': 0.05}),
                'weight_type': (WEIGHT_TYPES,),
                'combine_embeds': (['concat', 'add', 'subtract', 'average', 'norm average'],),
                'start_at': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.001}),
                'end_at': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.001}),
                'embeds_scaling': (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'],),
                'sharpening': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.05}),
                'layer_weights': ('STRING', {'default': '', 'multiline': True}),
            },
            'optional': {
                'image_negative': ('IMAGE',),
                'attn_mask': ('MASK',),
                'clip_vision': ('CLIP_VISION',),
                'insightface': ('INSIGHTFACE',),
            },
        }

    CATEGORY = _CATEGORY
    RETURN_TYPES = (
        'MODEL',
        'IMAGE',
        'MASK',
    )
    RETURN_NAMES = (
        'MODEL',
        'tiles',
        'masks',
    )

    def apply_ipadapter(self, model, ipadapter, image, weight, weight_faceidv2, weight_type, combine_embeds, start_at, end_at, embeds_scaling, layer_weights, sharpening, image_negative=None, attn_mask=None, clip_vision=None, insightface=None):
        # 1. Select the models
        if 'ipadapter' in ipadapter:
            ipadapter_model = ipadapter['ipadapter']['model']
            clip_vision = clip_vision if clip_vision is not None else ipadapter['clipvision']['model']
        else:
            ipadapter_model = ipadapter
            clip_vision = clip_vision

        if clip_vision is None:
            raise Exception('Missing CLIPVision model.')

        del ipadapter

        # 2. Extract the tiles
        tile_size = 256
        _, oh, ow, _ = image.shape
        if attn_mask is None:
            attn_mask = torch.ones([1, oh, ow], dtype=image.dtype, device=image.device)

        image = image.permute([0, 3, 1, 2])
        attn_mask = attn_mask.unsqueeze(1)
        attn_mask = T.Resize((oh, ow), interpolation=T.InterpolationMode.BICUBIC, antialias=True)(attn_mask)

        if oh / ow > 0.75 and oh / ow < 1.33:
            image = T.CenterCrop(min(oh, ow))(image)
            resize = (tile_size * 2, tile_size * 2)
            attn_mask = T.CenterCrop(min(oh, ow))(attn_mask)
        else:
            resize = (int(tile_size * ow / oh), tile_size) if oh < ow else (tile_size, int(tile_size * oh / ow))

        imgs = []
        for img in image:
            img = T.ToPILImage()(img)
            img = img.resize(resize, resample=Image.Resampling['LANCZOS'])
            imgs.append(T.ToTensor()(img))
        image = torch.stack(imgs)
        del imgs, img

        attn_mask = T.Resize(resize[::-1], interpolation=T.InterpolationMode.BICUBIC, antialias=True)(attn_mask)

        if oh / ow > 4 or oh / ow < 0.25:
            crop = (tile_size, tile_size * 4) if oh < ow else (tile_size * 4, tile_size)
            image = T.CenterCrop(crop)(image)
            attn_mask = T.CenterCrop(crop)(attn_mask)

        attn_mask = attn_mask.squeeze(1)

        if sharpening > 0:
            image = contrast_adaptive_sharpening(image, sharpening)

        image = image.permute([0, 2, 3, 1])

        _, oh, ow, _ = image.shape

        tiles_x = math.ceil(ow / tile_size)
        tiles_y = math.ceil(oh / tile_size)
        overlap_x = max(0, (tiles_x * tile_size - ow) / (tiles_x - 1 if tiles_x > 1 else 1))
        overlap_y = max(0, (tiles_y * tile_size - oh) / (tiles_y - 1 if tiles_y > 1 else 1))

        base_mask = torch.zeros([attn_mask.shape[0], oh, ow], dtype=image.dtype, device=image.device)

        tiles = []
        masks = []
        for y in range(tiles_y):
            for x in range(tiles_x):
                start_x = int(x * (tile_size - overlap_x))
                start_y = int(y * (tile_size - overlap_y))
                tiles.append(image[:, start_y : start_y + tile_size, start_x : start_x + tile_size, :])
                mask = base_mask.clone()
                mask[:, start_y : start_y + tile_size, start_x : start_x + tile_size] = attn_mask[:, start_y : start_y + tile_size, start_x : start_x + tile_size]
                masks.append(mask)
        del mask

        # 3. Apply the ipadapter to each group of tiles
        model = model.clone()
        for i in range(len(tiles)):
            ipa_args = {
                'image': tiles[i],
                'image_negative': image_negative,
                'weight': weight,
                'weight_faceidv2': weight_faceidv2,
                'weight_type': weight_type,
                'combine_embeds': combine_embeds,
                'start_at': start_at,
                'end_at': end_at,
                'attn_mask': masks[i],
                'unfold_batch': self.unfold_batch,
                'embeds_scaling': embeds_scaling,
                'insightface': insightface,
                'layer_weights': layer_weights,
            }
            model, _ = ipadapter_execute(model, ipadapter_model, clip_vision, **ipa_args)

        return (
            model,
            torch.cat(tiles),
            torch.cat(masks),
        )


IPADAPTER_CLASS_MAPPINGS = {
    'IPAdapterMSTiled-': IPAdapterMSTiled,
    'IPAdapterMSLayerWeights-': IPAdapterMSLayerWeights,
}

IPADAPTER_NAME_MAPPINGS = {
    'IPAdapterMSTiled-': 'IPAdapter MS Tiled',
    'IPAdapterMSLayerWeights-': 'IPAdapter MS Layer Weights',
}
