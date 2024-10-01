# ComfyUI_fnodes

ComfyUI_fnodes是一个为ComfyUI设计的自定义节点集合。这些节点提供了额外的功能,可以增强您的ComfyUI工作流程。

## 功能

- 掩码处理工具:
  - 组合掩码
  - 扩展/收缩掩码
  - 填充掩码中的孔洞
  - 反转掩码
  - 模糊掩码

- 图像处理工具
- IP-Adapter相关功能
- 其他杂项功能

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