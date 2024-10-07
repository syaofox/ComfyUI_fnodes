# ComfyUI_fnodes

ComfyUI_fnodes是一个为ComfyUI设计的自定义节点集合。这些节点提供了额外的功能,可以增强您的ComfyUI工作流程。

## 功能

- 文件操作工具:
  - ReadImage: 读取指定路径图片，返回图片和图片名称 [详细说明](docs/file_operations.md#readimage)
  - LoadImagesFromFolder: 读取文件夹中的图片，返回图片列表和图片批次 [详细说明](docs/file_operations.md#loadimagesfromfolder)
  - FilePathAnalyzer: 从文件路径中提取上层目录、文件名（不含扩展名）、扩展名和完整路径 [详细说明](docs/file_operations.md#filepathanalyzer)
  - RegexExtractor: 使用正则表达式从输入字符串中提取文本 [详细说明](docs/file_operations.md#regexextractor)
  - SelectFace: 选择人脸 [详细说明](docs/file_operations.md#selectface)

- 图像缩放工具:
  - GetImageSize: 获取图像的宽度、高度和数量 [详细说明](docs/image_scaling.md#getimagesize)
  - ImageScalerForSDModels: 根据SD模型类型将图像缩放到指定像素数 [详细说明](docs/image_scaling.md#imagescalerforsdmodels)
  - ImageScaleBySpecifiedSide: 根据指定边长缩放图片 [详细说明](docs/image_scaling.md#imagescalebyspecifiedside)
  - ComputeImageScaleRatio: 计算图像缩放比例和缩放后的宽高 [详细说明](docs/image_scaling.md#computeimagescaleratio)
  - ImageRotate: 旋转图像 [详细说明](docs/image_scaling.md#imagerotate)
  - TrimImageBorders: 去除图片黑边 [详细说明](docs/image_scaling.md#trimimageborders)
  
- IPAdapter工具:
  - IPAdapterMSTiled: 对图像进行分块处理并应用IPAdapter [详细说明](docs/ipadapter.md#ipadaptermstiledl)
  - IPAdapterMSLayerWeights: 为IPAdapter提供精细的层权重控制 [详细说明](docs/ipadapter.md#ipadaptermslayerweights)

- 图像处理工具:
  - ColorAdjustment: 对图片进行色彩校正 [详细说明](docs/image_processing.md#coloradjustment)
  - ColorTint: 应用图片颜色滤镜 [详细说明](docs/image_processing.md#colortint)
  - ColorBlockEffect: 实现图片色块化效果 [详细说明](docs/image_processing.md#colorblockeffect)
  - FlatteningEffect: 实现图片平面化效果 [详细说明](docs/image_processing.md#flatteningeffect)

- 遮罩工具:
  - OutlineMask: 给遮罩添加轮廓线 [详细说明](docs/mask_tools.md#outlinemask)
  - CreateBlurredEdgeMask: 根据指定图片创建模糊遮罩 [详细说明](docs/mask_tools.md#createblurrededgemask)
  - MaskChange: 修改和处理遮罩,支持扩展、填充和模糊等操作 [详细说明](docs/mask_tools.md#maskchange)
  - Depth2Mask: 将深度图像转换为遮罩 [详细说明](docs/mask_tools.md#depth2mask)

- 人脸分析工具:
  - GeneratePreciseFaceMask: 生成精确的人脸遮罩 [详细说明](docs/face_analysis.md#generateprecisefacemask)
  - AlignImageByFace: 根据图像中的人脸进行旋转对齐 [详细说明](docs/face_analysis.md#alignimagebyface)
  - FaceCutout: 切下人脸并进行缩放 [详细说明](docs/face_analysis.md#facecutout)
  - FacePaste: 将人脸图像贴回原图 [详细说明](docs/face_analysis.md#facepaste)
  - ExtractBoundingBox: 从边界框信息中提取坐标和尺寸 [详细说明](docs/face_analysis.md#extractboundingbox)
  - FaceMorph: 将人脸图像进行变形 [详细说明](docs/face_analysis.md#facemorph)

- 采样器工具:
  - ScheduleSamplerCustomTurbo: 自定义涡轮采样器,支持多次降噪 [详细说明](docs/schedule_samplers.md#schedulesamplercustomturbo)
  - ScheduleSamplerCustomAYS: 自定义AYS采样器,支持多次降噪和不同模型类型 [详细说明](docs/schedule_samplers.md#schedulesamplercustomays)
  - ScheduleSampler: 通用调度采样器,支持多次降噪 [详细说明](docs/schedule_samplers.md#schedulesampler)

- 杂项工具:
  - DisplayAny: 显示任何输入的字符串表示 [详细说明](docs/miscellaneous.md#displayany)
  - PrimitiveText: 创建一个基本的文本字符串 [详细说明](docs/miscellaneous.md#primitivetext)
  - FillMaskedImageArea: 使用指定的填充值填充图像中被遮罩覆盖的区域 [详细说明](docs/miscellaneous.md#fillmaskedimagearea)
  - Seed: 生成种子值 [详细说明](docs/miscellaneous.md#seed)

## 安装

1. 确保您已经安装了ComfyUI,部分功能需要安装其他自定义节点
    - ComfyUI_IPAdapter_plus: https://github.com/cubiq/ComfyUI_IPAdapter_plus
    - ComfyUI_FaceAnalysis: https://github.com/cubiq/ComfyUI_FaceAnalysis

3. 克隆此仓库到ComfyUI的`custom_nodes`目录:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/your-username/ComfyUI_fnodes.git
   ```
3. 安装依赖:
   ```bash
   pip install -r ComfyUI_fnodes/requirements.txt
   ```

## 使用

安装后,新的节点将在ComfyUI的节点列表中可用。您可以像使用其他节点一样将它们添加到您的工作流程中。

## 贡献

欢迎贡献!如果您有任何改进建议或发现了bug,请开启一个issue或提交一个pull request。

## 许可

本项目采用MIT许可证。详情请见[LICENSE](LICENSE)文件。