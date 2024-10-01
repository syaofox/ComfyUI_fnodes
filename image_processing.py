import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance

from comfy_extras.nodes_post_processing import Blend, Blur, Quantize

_CATEGORY = 'fnodes/image processing'


class ColorAdjustment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'temperature': (
                    'FLOAT',
                    {'default': 0, 'min': -100, 'max': 100, 'step': 5},
                ),
                'hue': ('FLOAT', {'default': 0, 'min': -90, 'max': 90, 'step': 5}),
                'brightness': (
                    'FLOAT',
                    {'default': 0, 'min': -100, 'max': 100, 'step': 5},
                ),
                'contrast': (
                    'FLOAT',
                    {'default': 0, 'min': -100, 'max': 100, 'step': 5},
                ),
                'saturation': (
                    'FLOAT',
                    {'default': 0, 'min': -100, 'max': 100, 'step': 5},
                ),
                'gamma': ('FLOAT', {'default': 1, 'min': 0.2, 'max': 2.2, 'step': 0.1}),
            },
        }

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'execute'
    CATEGORY = _CATEGORY
    DESCRIPTION = '对图片进行色彩校正'

    def execute(self, image: torch.Tensor, temperature: float, hue: float, brightness: float, contrast: float, saturation: float, gamma: float):
        batch_size, _, _, _ = image.shape
        result = torch.zeros_like(image)

        brightness /= 100
        contrast /= 100
        saturation /= 100
        temperature /= 100

        brightness = 1 + brightness
        contrast = 1 + contrast
        saturation = 1 + saturation

        for b in range(batch_size):
            tensor_image = image[b].numpy()

            modified_image = Image.fromarray((tensor_image * 255).astype(np.uint8))

            # brightness
            modified_image = ImageEnhance.Brightness(modified_image).enhance(brightness)

            # contrast
            modified_image = ImageEnhance.Contrast(modified_image).enhance(contrast)
            modified_image = np.array(modified_image).astype(np.float32)

            # temperature
            if temperature > 0:
                modified_image[:, :, 0] *= 1 + temperature
                modified_image[:, :, 1] *= 1 + temperature * 0.4
            elif temperature < 0:
                modified_image[:, :, 2] *= 1 - temperature
            modified_image = np.clip(modified_image, 0, 255) / 255

            # gamma
            modified_image = np.clip(np.power(modified_image, gamma), 0, 1)

            # saturation
            hls_img = cv2.cvtColor(modified_image, cv2.COLOR_RGB2HLS)
            hls_img[:, :, 2] = np.clip(saturation * hls_img[:, :, 2], 0, 1)
            modified_image = cv2.cvtColor(hls_img, cv2.COLOR_HLS2RGB) * 255

            # hue
            hsv_img = cv2.cvtColor(modified_image, cv2.COLOR_RGB2HSV)
            hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue) % 360
            modified_image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

            modified_image = modified_image.astype(np.uint8)
            modified_image = modified_image / 255
            modified_image = torch.from_numpy(modified_image).unsqueeze(0)
            result[b] = modified_image

        return (result,)


class ColorTint:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'strength': (
                    'FLOAT',
                    {'default': 1.0, 'min': 0.1, 'max': 1.0, 'step': 0.1},
                ),
                'mode': (
                    [
                        'sepia',
                        'red',
                        'green',
                        'blue',
                        'cyan',
                        'magenta',
                        'yellow',
                        'purple',
                        'orange',
                        'warm',
                        'cool',
                        'lime',
                        'navy',
                        'vintage',
                        'rose',
                        'teal',
                        'maroon',
                        'peach',
                        'lavender',
                        'olive',
                    ],
                ),
            },
        }

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'execute'
    CATEGORY = _CATEGORY
    DESCRIPTION = '图片颜色滤镜'

    def execute(self, image: torch.Tensor, strength: float, mode: str = 'sepia'):
        if strength == 0:
            return (image,)

        sepia_weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 1, 1, 3).to(image.device)

        mode_filters = {
            'sepia': torch.tensor([1.0, 0.8, 0.6]),
            'red': torch.tensor([1.0, 0.6, 0.6]),
            'green': torch.tensor([0.6, 1.0, 0.6]),
            'blue': torch.tensor([0.6, 0.8, 1.0]),
            'cyan': torch.tensor([0.6, 1.0, 1.0]),
            'magenta': torch.tensor([1.0, 0.6, 1.0]),
            'yellow': torch.tensor([1.0, 1.0, 0.6]),
            'purple': torch.tensor([0.8, 0.6, 1.0]),
            'orange': torch.tensor([1.0, 0.7, 0.3]),
            'warm': torch.tensor([1.0, 0.9, 0.7]),
            'cool': torch.tensor([0.7, 0.9, 1.0]),
            'lime': torch.tensor([0.7, 1.0, 0.3]),
            'navy': torch.tensor([0.3, 0.4, 0.7]),
            'vintage': torch.tensor([0.9, 0.85, 0.7]),
            'rose': torch.tensor([1.0, 0.8, 0.9]),
            'teal': torch.tensor([0.3, 0.8, 0.8]),
            'maroon': torch.tensor([0.7, 0.3, 0.5]),
            'peach': torch.tensor([1.0, 0.8, 0.6]),
            'lavender': torch.tensor([0.8, 0.6, 1.0]),
            'olive': torch.tensor([0.6, 0.7, 0.4]),
        }

        scale_filter = mode_filters[mode].view(1, 1, 1, 3).to(image.device)

        grayscale = torch.sum(image * sepia_weights, dim=-1, keepdim=True)
        tinted = grayscale * scale_filter

        result = tinted * strength + image * (1 - strength)
        return (result,)


class ColorBlockEffect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'strength': (
                    'INT',
                    {'default': 10, 'min': 1, 'max': 100, 'step': 1},
                ),
            },
        }

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'execute'
    CATEGORY = _CATEGORY
    DESCRIPTION = '图片色块化'

    def execute(self, image: torch.Tensor, strength: int):
        color_correct = ColorAdjustment()
        blur = Blur()
        quantize_node = Quantize()
        blender = Blend()

        blurred_image = blur.blur(image, blur_radius=strength, sigma=1.0)
        blurred_image = torch.cat(blurred_image, dim=1)

        quantized_image = quantize_node.quantize(blurred_image, colors=5, dither='bayer-2')
        quantized_image = torch.cat(quantized_image, dim=1)

        color_corrected_image = color_correct.execute(quantized_image, temperature=0, hue=0, brightness=5, contrast=0, saturation=-100, gamma=1)
        color_corrected_image = torch.cat(color_corrected_image, dim=1)

        blender_image = blender.blend_images(color_corrected_image, image, blend_factor=1, blend_mode='overlay')
        blender_image = torch.cat(blender_image, dim=1)

        flat_image = color_correct.execute(blender_image, temperature=0, hue=0, brightness=5, contrast=5, saturation=50, gamma=1.2)
        flat_image = torch.cat(flat_image, dim=1)
        return (flat_image,)


IMAGE_PROCESSING_CLASS_MAPPINGS = {
    'ColorAdjustment-': ColorAdjustment,
    'ColorTint-': ColorTint,
    'ColorBlockEffect-': ColorBlockEffect,
}

IMAGE_PROCESSING_NAME_MAPPINGS = {
    'ColorAdjustment-': 'Image Color Adjustment',
    'ColorTint-': 'Image Color Tint',
    'ColorBlockEffect-': 'Image Color Block Effect',
}
