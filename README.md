# ComfyUI_fnodes

ComfyUI_fnodes是一个为ComfyUI设计的自定义节点集合。这些节点提供了额外的功能,可以增强您的ComfyUI工作流程。

## 功能

- 文件操作工具:
  - ReadImage: 读取指定路径图片，返回图片和图片名称
  - LoadImagesFromFolder: 读取文件夹中的图片，返回图片列表和图片批次
  - FilePathAnalyzer: 从文件路径中提取上层目录、文件名（不含扩展名）、扩展名和完整路径
  - RegexExtractor: 使用正则表达式从输入字符串中提取文本
  - SelectFace: 选择人脸

- 图像缩放工具:
  - GetImageSize: 获取图像的宽度、高度和数量
  - ImageScalerForSDModels: 根据SD模型类型将图像缩放到指定像素数
  - ImageScaleBySpecifiedSide: 根据指定边长缩放图片
  - ComputeImageScaleRatio: 计算图像缩放比例和缩放后的宽高
  - ImageRotate: 旋转图像
  - TrimImageBorders: 去除图片黑边
  
- IPAdapter工具:
  - IPAdapterMSTiled: 对图像进行分块处理并应用IPAdapter
  - IPAdapterMSLayerWeights: 为IPAdapter提供精细的层权重控制

- 图像处理工具:
  - ColorAdjustment: 对图片进行色彩校正
  - ColorTint: 应用图片颜色滤镜
  - ColorBlockEffect: 实现图片色块化效果
  - FlatteningEffect: 实现图片平面化效果


- 遮罩工具:
  - OutlineMask: 给遮罩添加轮廓线
  - CreateBlurredEdgeMask: 根据指定图片创建模糊遮罩

- 杂项工具:
  - DisplayAny: 显示任何输入的字符串表示
  - PrimitiveText: 创建一个基本的文本字符串
  - FillMaskedImageArea: 使用指定的填充值填充图像中被遮罩覆盖的区域

- 人脸分析工具:
  - GeneratePreciseFaceMask: 生成精确的人脸遮罩
  - AlignImageByFace: 根据图像中的人脸进行旋转对齐
  - FaceCutout: 切下人脸并进行缩放

## 安装

1. 确保您已经安装了ComfyUI。
2. 克隆此仓库到ComfyUI的`custom_nodes`目录:
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