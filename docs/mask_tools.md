# 遮罩工具

## OutlineMask
给遮罩添加轮廓线。

### 输入
- `mask`: 输入遮罩
- `outline_width`: 轮廓线宽度
- `tapered_corners`: 是否使用渐变角落

### 输出
- `mask`: 添加轮廓线后的遮罩

## CreateBlurredEdgeMask
根据指定图片或尺寸创建模糊边缘遮罩。

### 输入
- `width`: 宽度
- `height`: 高度
- `border`: 边框宽度
- `border_percent`: 边框宽度百分比
- `blur_radius`: 模糊半径
- `blur_radius_percent`: 模糊半径百分比
- `image` (可选): 输入图像

### 输出
- `mask`: 创建的模糊边缘遮罩

## MaskChange
修改和处理遮罩。

### 输入
- `mask`: 输入遮罩
- `grow`: 扩展/收缩像素数
- `grow_percent`: 扩展/收缩百分比
- `grow_tapered`: 是否使用渐变扩展
- `blur`: 模糊半径
- `fill`: 是否填充孔洞

### 输出
- `mask`: 处理后的遮罩
- `inverted_mask`: 处理后的反转遮罩