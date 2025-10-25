# 安装指南

## 🐳 使用 Docker 安装

我们提供了预构建的 Docker 镜像以便快速开始。请从[镜像地址](https://alibaba.github.io/ROLL/docs/English/QuickStart/image_address)中选择您需要的镜像。

## 🛠️ 在自定义环境中安装

如果我们的预构建 Docker 镜像与您的环境不兼容，您可以在您的 Python 环境中安装 ROLL 及其依赖项。请确保您满足以下先决条件：

```bash
# 先决条件
CUDA 版本 >= 12.4
cuDNN 版本 >= 9.1.0
PyTorch >= 2.5.1
SGlang >= 0.4.3
vLLM >= 0.7.3

# 克隆仓库并安装
git clone https://github.com/alibaba/ROLL.git
cd ROLL
pip install -r requirements.txt # 或按照您的特定安装步骤
# 对于开发，可以考虑使用：pip install -e .
```
