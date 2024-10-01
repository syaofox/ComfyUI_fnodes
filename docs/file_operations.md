# 文件操作工具

## ReadImage
读取指定路径图片，返回图片和图片名称。

### 输入
- `image_path`: 图片文件的路径（STRING，默认值：'images'）

### 输出
- `IMAGE`: 读取的图片
- `STRING`: 图片文件名（不含扩展名）

## LoadImagesFromFolder
读取文件夹中的图片，返回图片列表和图片批次。

### 输入
- `input_path`: 包含图片的文件夹路径（STRING，默认值：''）
- `start_index`: 起始索引（INT，默认值：0，范围：0-9999）
- `max_index`: 最大索引（INT，默认值：1，范围：1-9999）

### 输出
- `IMAGE`: 读取的图片列表
- `IMAGE`: 图片批次

## FilePathAnalyzer
从文件路径中提取上层目录、文件名（不含扩展名）、扩展名和完整路径。

### 输入
- `file_path`: 文件的完整路径（STRING，默认值：'file.txt'）

### 输出
- `STRING`: 上层目录
- `STRING`: 文件名（不含扩展名）
- `STRING`: 文件扩展名
- `STRING`: 完整文件路径

## RegexExtractor
使用正则表达式从输入字符串中提取文本。

### 输入
- `input_string`: 要进行匹配的输入字符串（STRING，默认值：''）
- `regex_pattern`: 正则表达式模式（STRING，默认值：''）
- `group_number`: 要提取的组号（INT，默认值：0，范围：0-100）

### 输出
- `STRING`: 提取的文本

## SelectFace
选择人脸。

### 输入
- `face_name`: 人脸名称（从预定义列表中选择）

### 输出
- `STRING`: 人脸路径
- `STRING`: 人脸名称
