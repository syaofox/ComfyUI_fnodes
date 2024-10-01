# 图像缩放工具

## GetImageSize
获取图像的宽度、高度和数量。

### 输入
- `image`: 输入图像

### 输出
- `width`: 图像宽度
- `height`: 图像高度
- `count`: 图像数量

## ImageScalerForSDModels
根据SD模型类型将图像缩放到指定像素数。

### 输入
- `image`: 输入图像
- `upscale_method`: 上采样方法
- `sd_model_type`: SD模型类型
- `mask`: (可选) 输入掩码

### 输出
- `image`: 缩放后的图像
- `mask`: 缩放后的掩码
- `width`: 缩放后的宽度
- `height`: 缩放后的高度
- `min_dimension`: 缩放后的最小尺寸

### 描述
根据SD模型类型缩放图片到指定像素数，sd15为512x512，sd15+为512x768，sdxl为1024x1024，sdxl+为1024x1280

## ImageScaleBySpecifiedSide
根据指定边长缩放图片。

### 输入
- `image`: 输入图像
- `upscale_method`: 上采样方法
- `size`: 指定边长
- `shorter`: 是否参照短边
- `mask`: (可选) 输入掩码

### 输出
- `image`: 缩放后的图像
- `mask`: 缩放后的掩码
- `width`: 缩放后的宽度
- `height`: 缩放后的高度
- `min_dimension`: 缩放后的最小尺寸

### 描述
根据指定边长缩放图片，shorter为True时参照短边，否则参照长边

## ComputeImageScaleRatio
计算图像缩放比例和缩放后的宽高。

### 输入
- `image`: 输入图像
- `target_max_size`: 目标最大尺寸

### 输出
- `rescale_ratio`: 缩放比例
- `width`: 缩放后的宽度
- `height`: 缩放后的高度

### 描述
根据引用图片的大小和目标最大尺寸，返回缩放比例和缩放后的宽高

## ImageRotate
旋转图像。

### 输入
- `image_from`: 输入图像
- `angle`: 旋转角度
- `expand`: 是否扩展图像尺寸

### 输出
- `rotated_image`: 旋转后的图像

## TrimImageBorders
去除图片黑边。

### 输入
- `image`: 输入图像
- `threshold`: 黑边阈值

### 输出
- `image`: 去除黑边后的图像

### 描述
图片去黑边