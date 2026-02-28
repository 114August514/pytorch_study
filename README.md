# PyTorch Learning Project

本项目是一个用于学习 PyTorch 和《动手学深度学习》(D2L) 的练习仓库。包含了多个单元的 Jupyter Notebook 练习以及相关的数据处理脚本。

## 环境配置指南

为了确保项目依赖环境的隔离与稳定，**强烈建议用户自行创建并管理虚拟环境**。本项目推荐使用 [uv](https://github.com/astral-sh/uv) 进行配置。

### 1. 使用 uv 配置环境

`uv` 是一个极快的 Python 包管理器，能够根据 [pyproject.toml](pyproject.toml) 自动创建、同步虚拟环境。

> [!CAUTION]
> **Windows 用户注意：**
> 本项目的 [pyproject.toml](pyproject.toml) 已锁定为 **Linux** 环境 (`sys_platform == 'linux'`)。**在 Windows 原生命令行中直接运行 `uv sync` 将报错**。Windows 用户推荐改用 **WSL2**。

```bash
# 1. 安装 uv (如果尚未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 同步环境并安装依赖 (Windows 环境下执行会有问题, 比较推荐重新搭建虚拟环境)
uv sync

# 3. 激活环境
source .venv/bin/activate

# (可选) 添加新依赖，会自动持久化到 pyproject.toml
uv add <package_name>
```

### 2. 特别说明：GPU (CUDA) 支持

[pyproject.toml](pyproject.toml) 中的 `torch` 默认可能会安装 CPU 版本。若需启用 GPU 加速，请根据你的 CUDA 版本从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取对应的安装命令。

## 项目结构

- `test_unit*.ipynb`: 各章节的练习笔记。
- [d2l_utils.py](d2l_utils.py): 自定义的辅助函数工具包。
- `temp/data/`: 存放实验所需的数据集（如 FashionMNIST, Kaggle House Price 等）。

## 核心开发规范

为了提升代码的可维护性与复现性，本项目遵循以下规范：

- **强类型检查**: 所有自定义函数必须包含 Python Type Hints (PEP 484)。
- **文档规范**: 核心工具函数遵循 [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) 注释规范。
- **模块化设计**: 
    - 实验性探索在 `test_unit*.ipynb` 中完成。
    - 稳定后的算子、训练引擎及评估逻辑将沉淀至 `d2l_utils.py`。

## 验证环境

安装完成后，可以运行以下命令检查 PyTorch 是否能识别 GPU：

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```
