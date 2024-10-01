# 采样器工具

## ScheduleSamplerCustomTurbo

ScheduleSamplerCustomTurbo是一个自定义的涡轮采样器,支持多次降噪过程。

### 参数

- `model`: 用于去噪的模型
- `add_noise`: 是否添加噪声 (布尔值)
- `noise_seed`: 噪声种子 (整数)
- `cfg`: Classifier-Free Guidance比例 (浮点数)
- `positive`: 正面提示词条件
- `negative`: 负面提示词条件
- `sampler_name`: 采样器名称
- `steps`: 采样步数 (整数,1-10)
- `denoise_schedule`: 降噪计划 (字符串,例如 "0.5,0.25")
- `latent_image`: 潜在图像

### 输出

- `output`: 采样后的潜在图像
- `denoised_output`: 去噪后的潜在图像

## ScheduleSamplerCustomAYS

ScheduleSamplerCustomAYS是一个自定义的AYS采样器,支持多次降噪和不同的模型类型。

### 参数

- `model`: 用于去噪的模型
- `add_noise`: 是否添加噪声 (布尔值)
- `noise_seed`: 噪声种子 (整数)
- `cfg`: Classifier-Free Guidance比例 (浮点数)
- `positive`: 正面提示词条件
- `negative`: 负面提示词条件
- `sampler_name`: 采样器名称
- `model_type`: 模型类型 (SD1, SDXL, 或 SVD)
- `steps`: 采样步数 (整数,10-10000)
- `denoise_schedule`: 降噪计划 (字符串,例如 "0.5,0.25")
- `latent_image`: 潜在图像

### 输出

- `output`: 采样后的潜在图像
- `denoised_output`: 去噪后的潜在图像

## ScheduleSampler

ScheduleSampler是一个通用的调度采样器,支持多次降噪过程。

### 参数

- `model`: 用于去噪的模型
- `seed`: 随机种子 (整数)
- `steps`: 采样步数 (整数,1-10000)
- `cfg`: Classifier-Free Guidance比例 (浮点数)
- `sampler_name`: 采样器名称
- `scheduler`: 调度器名称
- `positive`: 正面提示词条件
- `negative`: 负面提示词条件
- `latent_image`: 潜在图像
- `denoise_schedule`: 降噪计划 (字符串,例如 "0.5,0.25")

### 输出

- `latent`: 采样后的潜在图像

## 使用说明

这些采样器节点允许您对生成过程进行更精细的控制。通过调整降噪计划,您可以在单次采样过程中实现多次降噪,potentially提高生成图像的质量或实现特殊效果。

例如,使用 "0.5,0.25" 作为降噪计划意味着首先对图像进行50%的降噪,然后再进行25%的降噪。这种方法可能有助于在保持某些细节的同时改善整体图像质量。

请根据您的具体需求和实验结果来调整这些参数.
