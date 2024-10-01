# IPAdapter工具

## IPAdapterMSLayerWeights
为IPAdapter提供精细的层权重控制。

### 输入
- `model_type`: 模型类型,可选 'SDXL' 或 'SD15'
- `L0` 到 `L15`: 各层的权重值,范围 0.0-10.0

### 输出
- `layer_weights`: 包含层权重的字符串

## IPAdapterMSTiled
对图像进行分块处理并应用IPAdapter。

### 输入
- `model`: 模型
- `ipadapter`: IPAdapter模型
- `image`: 输入图像
- `weight`: IPAdapter权重,默认1.0,范围-1到5
- `weight_faceidv2`: FaceID v2权重,默认1.0,范围-1到5
- `weight_type`: 权重类型
- `combine_embeds`: 嵌入组合方式,可选 'concat', 'add', 'subtract', 'average', 'norm average'
- `start_at`: 开始位置,默认0.0,范围0-1
- `end_at`: 结束位置,默认1.0,范围0-1
- `embeds_scaling`: 嵌入缩放方式,可选 'V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'
- `sharpening`: 锐化程度,默认0.0,范围0-1
- `layer_weights`: 层权重字符串
- `image_negative`: 负面图像(可选)
- `attn_mask`: 注意力掩码(可选)
- `clip_vision`: CLIP视觉模型(可选)
- `insightface`: InsightFace模型(可选)

### 输出
- `MODEL`: 处理后的模型
- `tiles`: 处理后的图像块
- `masks`: 处理后的掩码