# 遮罩工具

## OutlineMask
给遮罩添加轮廓线。

### 输入
- `mask`: 输入遮罩
- `thickness`: 轮廓线粗细

### 输出
- `outlined_mask`: 添加轮廓线后的遮罩

## CreateBlurredEdgeMask
根据指定图片创建模糊遮罩。

### 输入
- `image`: 输入图像
- `blur_radius`: 模糊半径

### 输出
- `blurred_mask`: 创建的模糊遮罩

## MaskToImage
将遮罩转换为图像。

### 输入
- `mask`: 输入遮罩

### 输出
- `image`: 转换后的图像

## ImageToMask
将图像转换为遮罩。

### 输入
- `image`: 输入图像

### 输出
- `mask`: 转换后的遮罩