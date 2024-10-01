# 人脸分析工具

## GeneratePreciseFaceMask
生成精确的人脸遮罩。

### 输入
- `input_image`: 输入图像
- `grow`: 遮罩扩展像素数 (可选,默认0)
- `grow_percent`: 遮罩扩展百分比 (可选,默认0.00)
- `grow_tapered`: 是否使用锥形扩展 (可选,默认False)
- `blur`: 遮罩模糊半径 (可选,默认0)
- `fill`: 是否填充遮罩孔洞 (可选,默认False)

### 输出
- `mask`: 生成的人脸遮罩
- `inverted_mask`: 反转的人脸遮罩
- `image`: 应用遮罩后的图像

## AlignImageByFace
根据图像中的人脸进行旋转对齐。

### 输入
- `analysis_models`: 分析模型
- `image_from`: 要对齐的输入图像
- `expand`: 是否扩展图像以包含整个旋转后的图像 (默认True)
- `simple_angle`: 是否使用简化角度 (默认False)
- `image_to`: 目标对齐图像 (可选)

### 输出
- `aligned_image`: 对齐后的图像
- `rotation_angle`: 旋转角度
- `inverse_rotation_angle`: 反向旋转角度

## FaceCutout
切下人脸并进行缩放。

### 输入
- `analysis_models`: 分析模型
- `image`: 输入图像
- `padding`: 额外填充像素数
- `padding_percent`: 额外填充百分比
- `face_index`: 要处理的人脸索引 (默认-1,表示第一个检测到的人脸)
- `rescale_mode`: 缩放模式 ('sdxl', 'sd15', 'sdxl+', 'sd15+', 'none', 'custom')
- `custom_megapixels`: 自定义目标大小 (以百万像素为单位)

### 输出
- `cutout_image`: 切下并缩放的人脸图像
- `bounding_info`: 边界框信息

## FacePaste
将人脸图像贴回原图。

### 输入
- `destination`: 目标图像
- `source`: 源人脸图像
- `bounding_info`: 边界框信息
- `margin`: 额外边缘像素数
- `margin_percent`: 额外边缘百分比
- `blur_radius`: 边缘模糊半径

### 输出
- `image`: 贴回人脸后的图像
- `mask`: 应用的遮罩