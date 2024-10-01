import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

_CATEGORY = 'fnodes/files'


class ReadImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image_path': ('STRING', {'default': 'images'}),
            }
        }

    RETURN_TYPES = ('IMAGE', 'STRING')
    RETURN_NAMES = ('image', 'file_stem')
    FUNCTION = 'execute'
    CATEGORY = _CATEGORY
    DESCRIPTION = '读取指定路径图片，返回图片和图片名称'

    def execute(self, image_path):
        # 去掉可能存在的双引号
        image_path = image_path.strip('"')

        if not os.path.exists(image_path):
            raise FileNotFoundError(f'文件未找到: {image_path}')

        file_stem = str(Path(image_path).stem)

        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)

        if img is None:
            raise ValueError(f'无法从文件中读取有效图像: {image_path}')

        if img.mode == 'I':
            img = img.point(lambda i: i * (1 / 255))
        img = img.convert('RGB')

        image = np.array(img).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return (image, file_stem)


class FilePathAnalyzer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'file_path': ('STRING', {'default': 'file.txt'}),
            }
        }

    RETURN_TYPES = ('STRING', 'STRING', 'STRING', 'STRING')
    RETURN_NAMES = ('parent_dir', 'file_stem', 'file_extension', 'full_path')
    FUNCTION = 'execute'
    CATEGORY = _CATEGORY
    DESCRIPTION = '从文件路径中提取上层目录、文件名（不含扩展名）、扩展名和完整路径'

    def execute(self, file_path):
        # 去掉可能存在的双引号
        file_path = file_path.strip('"')
        path = Path(file_path)

        parent_dir = str(path.parent)
        file_stem = path.stem
        file_extension = path.suffix
        full_path = str(path.absolute())

        return (parent_dir, file_stem, file_extension, full_path)


class RegexExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'input_string': ('STRING', {'default': ''}),
                'regex_pattern': ('STRING', {'default': ''}),
                'group_number': ('INT', {'default': 0, 'min': 0, 'max': 100}),
            }
        }

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('extracted_text',)
    FUNCTION = 'execute'
    CATEGORY = _CATEGORY
    DESCRIPTION = '使用正则表达式从输入字符串中提取文本'

    def execute(self, input_string, regex_pattern, group_number):
        try:
            match = re.search(regex_pattern, input_string)
            if match:
                groups = match.groups()
                if 0 <= group_number <= len(groups):
                    return (match.group(group_number),)
                else:
                    return ('组号超出范围',)
            else:
                return ('未找到匹配',)
        except re.error:
            return ('无效的正则表达式',)


FILE_CLASS_MAPPINGS = {
    'ReadImage-': ReadImage,
    'FilePathAnalyzer-': FilePathAnalyzer,
    'RegexExtractor-': RegexExtractor,
}

FILE_NAME_MAPPINGS = {
    'ReadImage-': 'Read Image from Path',
    'FilePathAnalyzer-': 'FilePath Analyzer',
    'RegexExtractor-': 'Regex Extractor',
}
