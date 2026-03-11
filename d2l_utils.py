#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
D2L Utilities for PyTorch Learning.

该模块集成了《动手学深度学习》课程中的核心工具函数，包括数据加载、
模型训练循环、评估指标计算以及可视化辅助工具。
所有函数均采用强类型标注，并遵循 Google Python 代码规范。

使用原则：仅在实现章节之后才会调用封装的函数。

Usage:
    import d2l_utils as d2l
    train_iter, test_iter = d2l.load_
"""

from __future__ import annotations # 允许在类型标注中使用尚未定义的类 (Python 3.7+)

# 元数据声明
__author__ = "114August514"
__version__ = "0.1.0"
__status__ = "Development"

# --- 标准库导入 ---
import sys
from typing import (
    Final,
    Callable,
    Iterable,
    Optional,
    Union,
    Iterator
)
from pathlib import Path

# --- 第三方库导入 ---
import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn, Tensor
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms

# --- 1. 路径常量 (Path Constants) ---
# 获取当前工具库文件所在的绝对路径
# 这保证了无论在哪个文件夹运行 notebook, 路径永远以 d2l_utils.py 为准
BASE_DIR: Final[Path] = Path(__file__).resolve().parent

# 定义数据存放目录
# 默认在 BASE_DIR / temp / data
DATA_ROOT: Final[Path] = BASE_DIR / "temp" / "data"

# --- 2. 系统环境常量 (System Constants) ---
# 自动判断当前系统是否为 Windows
IS_WINDOWS: Final[bool] = sys.platform.startswith('win')

# --- 3. 默认超参数 (可选，作为默认建议值) ---
DEFAULT_BATCH_SIZE: Final[int] = 256
DEFAULT_LEARNING_RATE: Final[float] = 0.1

# --- 4. 显式导出声明 ---
# 定义 __all__ 可以控制 from d2l_utils import * 时导出的内容
__all__ = [
    'DATA_ROOT',
    'IS_WINDOWS',
    'Accumulator',
    'get_default_device',
    'load_fashion_mnist',
    'count_correct',
    'evaluate_accuracy',
    'evaluate_accuracy_gpu',
    'cross_entropy',
    'sgd',
    'train_softmax',
    'DataStreamVisualizer'
]

# ==============================================================================
# SECTION 1: 基础工具类与辅助函数 (Basic Utilities & Helpers)
# 作用：通用的“螺丝钉”，被后面的训练逻辑反复调用。
# ==============================================================================
# 累加器
# 一个用于在多个批次（Batch）间累加数据（如损失总和、正确个数）的小工具。
class Accumulator:
    """在 n 个变量上累加数值的实用工具。"""
    def __init__(self, n: int):
        """初始化 n 个变量，初始值均为 0.0。
        
        Args:
            n: 需要累加的变量个数。
        """
        self.data: list[float] = [0.0] * n

    def add(self, *args: float) -> None:
        """将传入的多个数值分别累加到相应的位置。
        
        Args:
            *args: 与初始化时 n 长度一致的数值序列。
        """
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self) -> None:
        """重置所有累加值为 0。"""
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx: int) -> float:
        """支持通过索引访问，例如 acc[0]。"""
        return self.data[idx]

def get_dataloader_workers() -> int:
    """根据系统常量返回推荐的进程数。"""
    return 0 if IS_WINDOWS else 4

def get_default_device() -> torch.device:
    """获取当前环境的最佳计算设备 (暂未考虑多个 GPU 情况，即仅简单考虑了硬件类型)。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
        # cuda 与 cuda:0 是等价的
        # 对于第二张显卡，用的是 torch.device("cuda:1")
    
    # 如果是 Mac M1/M2/M3，可以使用 mps
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    
    return torch.device("cpu")

# ==============================================================================
# SECTION 2: 数据流水线 (Data Pipeline)
# 作用：处理数据从 硬盘 -> 内存 -> DataLoader 的过程。
# 逻辑顺序：下载 -> Dataset -> DataLoader
# ==============================================================================
def download_fashion_mnist(root: Path = DATA_ROOT) -> None:
    """确保数据集已下载到本地。
    
    此函数是幂等的 (多次调用不产生副作用)。

    Args:
        root: 数据存放路径。
    """
    # 仅下载，不进行任何变换
    # root: 数据存放路径
    # train: True 表示训练集，False 表示测试集
    torchvision.datasets.FashionMNIST(root=str(root), train=True, download=True)
    torchvision.datasets.FashionMNIST(root=str(root), train=False, download=True)
    print(f"Dataset check complete at: {root.absolute()}")

# 预处理数据
# 原始图片 (-> 缩放 (Resize)) -> 转为张量 (ToTensor) -> 神经网络。
def get_fashion_mnist_dataset(
        root: Path = DATA_ROOT,
        resize: Optional[int] = None
) -> tuple[data.Dataset, data.Dataset]:
    """获取预处理后的数据集对象。
    
    Args:
        root: 数据存放路径。
        resize: 是否调整图像尺寸。
        
    Returns:
        (train_dataset, test_dataset)
    """
    # 1. 定义预处理流程 (Transforms)
    # transforms.ToTensor() 做两件事：
    #   a. 改变形状：从 (H, W, C) 变为 (C, H, W) -> 28x28x1 变为 1x28x28
    #   b. 归一化：将像素值从 [0, 255] 映射到 [0, 1] 的 float32
    trans: list[Callable] = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))

    # 组合多个预处理步骤
    trans_compose = transforms.Compose(trans)

    # 2. 加载数据并进行变换 (预处理)
    # 如果 download_fashion_mnist 没跑成功，这里会直接报错（因为 download = False）
    train_ds = torchvision.datasets.FashionMNIST(root=str(root), train=True, transform=trans_compose, download=False) 
    test_ds = torchvision.datasets.FashionMNIST(root=str(root), train=False, transform=trans_compose, download=False) 

    return train_ds, test_ds

# 生成迭代器
def load_fashion_mnist(
        batch_size: int = DEFAULT_BATCH_SIZE,
        resize: Optional[int] = None,
        num_workers: int = get_dataloader_workers()
) -> tuple[data.dataloader, data.dataloader]:
    """整合下载、预处理与加载逻辑，返回最终的 DataLoader。

    这是供模型训练脚本调用的主入口函数。
    支持可选的图像尺寸调整，并默认使用多进程加速数据读取。
    
    Args: 
        batch_size: 每个训练批次的样本数量。
        resize: 可选参数。如果提供，图像将通过插值算法缩放为 (resize, resize) 形状。
        num_workers: 用于数据加载的子进程数量。在 Linux 环境下增加此值可加速读取。

    Returns:
        一个包含两个元素的元组。
        - train_iter (data.DataLoader): 训练数据集的迭代器，已开启随机打乱。
        - test_iter (data.DataLoader): 测试数据集的迭代器，不开启随机打乱。
    """
    # 1. 确保原始数据在硬盘上
    download_fashion_mnist()

    # 2. 获取经过预处理的数据集对象
    train_ds, test_ds = get_fashion_mnist_dataset(resize=resize)

    # 3. 创建高效的数据迭代器
    # shuffle=True: 在每个 epoch 开始时打乱数据，防止模型学到样本的先后顺序（偏见）
    # pin_memory=True: (可选) 如果使用 GPU，开启此项可以加快张量从内存复制到显存的速度
    train_iter = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_iter = data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_iter, test_iter

# ==============================================================================
# SECTION 3: 核心算法组件 (Core Algorithms)
# 作用：定义损失函数、优化器逻辑、指标计算。
# ==============================================================================
def count_correct(y_hat: Tensor, y: Tensor) -> int:
    """计算预测正确的数量。
    
    Args:
        y_hat: 预测概率分布。
        y: 真实标签。

    Returns:
        这一批次中预测正确的样本总数（整数）。
    """
    # 鲁棒性：
    # - 判断 y_hat 是模型输出的原始概率，还是预测值。
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 找到概率最大的那个类别的索引
        y_hat = y_hat.argmax(axis=1)
    
    # 2. 将预测结果转为与真实标签相同的数据类型，防止比较报错
    # 3. 比较预测值与真实值，得到布尔张量 (True 为对，False 为错)
    # 4. 将布尔值转为整数并求和，最后用 .item() 转为 Python 原生整数
    # 5. int() 在最后进行显示转换。
    cmp = y_hat.type(y.dtype) == y
    return int(cmp.type(y.dtype).sum().item())

def cross_entropy(y_hat: Tensor, y: Tensor) -> Tensor:
    """计算交叉熵损失。
    
    Args:
        y_hat: 预测概率分布矩阵，形状为 (batch_size, num_classes)。
        y: 真实标签向量，形状为 (batch_size,)。

    Returns:
        包含每个样本损失的一维张量。
    """
    # 1. y_hat[range(len(y_hat)), y] 是 PyTorch 的高级索引:
    #    - range(len(y_hat)) 产生行索引 [0, 1, 2, ...]
    #    - y 提供了列索引 (即正确所在的列)
    #    - 结果是从每一行中挑出正确类别的那个概率值
    correct_class_probabilities: Tensor = y_hat[range(len(y_hat)), y]

    # 2. 取负对数
    return -torch.log(correct_class_probabilities)

def sgd(params: Iterable[Tensor], lr: float, batch_size: int) -> None:
    """小批量随机梯度下降 (SGD) 更新参数。
    
    Args:
        params: 更新的参数。
        lr: 学习率 (步长)。
        batch_size: 批数据规模 (用于对总梯度取平均)。
    """
    # 1. 更新参数时不需要追踪计算图，使用 no_grad 节省资源
    with torch.no_grad():
        for param in params:
            # 2. 执行梯度下降更新: w = w - lr * (grad / batch_size)
            # 注意：这里使用原地操作 -= 节省内存
            param -= lr * param.grad / batch_size

            # 3. 核心：手动清空梯度。
            # PyTorch 默认累加梯度，如果不清零，下一批次的计算会出错。
            param.grad.zero_()

# ==============================================================================
# SECTION 4: 训练与评估流程 (Training & Evaluation Flow)
# 作用：将数据和算法组合起来，这是整个库的“引擎”。
# 逻辑顺序：评估函数 -> 单轮训练 -> 总训练循环
# ==============================================================================
# 评估函数
def evaluate_accuracy(net: Callable[[Tensor], Tensor], data_iter: DataLoader) -> float:
    """计算在指定数据集上模型的准确率。
    Args:
        net: 神经网络模型。
        data_iter: 指定的数据集。

    Returns:
        计算得到的模型准确率。
    """
    if isinstance(net, nn.Module):
        net.eval() # 设置为评估模式

    metric = Accumulator(2) # [正确预测数, 样本总数]

    with torch.no_grad(): # 评估时不需要计算梯度，节省显存和计算资源
        for X, y in data_iter:
            metric.add(count_correct(net(X), y), y.numel())

    return metric[0] / metric[1]

# 评估函数 (可以接受 GPU)
def evaluate_accuracy_gpu(net: Callable[[Tensor], Tensor], data_iter: data.DataLoader, device: torch.device = None) -> float:
    """使用 GPU 计算在指定数据集上模型的准确率。
    
    Args:
        net: 神经网络模型。
        data_iter: 验证集或测试集的迭代器。
        device: 指定运行的设备。如果为 None，则尝试从模型参数中推断。

    Returns:
        float: 计算得到的模型准确率。
    """
    if isinstance(net, nn.Module):
        net.eval() # 设置为评估模式
        # 如果未指定 device，则取模型第一个参数所在的设备
        if not device:
            try:
                device = next(iter(net.parameters())).device
            except StopIteration:
                # 如果模型没有参数
                device = torch.device('cpu')

    # Accumulator 是我们在 d2l_utils 中定义的累加器
    # [正确预测数, 样本总数]
    metric = Accumulator(2)

    with torch.no_grad(): # 评估时不需要计算梯度，节省显存和计算资源
        for X, y in data_iter:
            # 将数据搬运到与模型相同的设备
            X, y = X.to(device), y.to(device)

            # 计算正确数并累加
            metric.add(count_correct(net(X), y), y.numel())

    return metric[0] / metric[1]

# 单轮训练函数
def train_softmax_epoch(
    net: Callable[[Tensor], Tensor], # 因为现在是从零实现，所以没有写成 nn.Module
    train_iter: DataLoader,
    loss: Callable[[Tensor, Tensor], Tensor], # 因为现在是从零实现，所以没有写成 nn.modules.loss._Loss
    updater: Union[torch.optim.Optimizer, Callable] # 二选一，可以选择官方的随机梯度下降，也可以考虑自己写的 sgd（但它在隔壁文件）
) -> tuple[float, float]:
    """训练模型一个迭代周期(Epoch)。

    Args:
        net: 神经网络模型。
        train_iter: 训练数据迭代器。
        loss: 损失函数。
        updater: 优化器 (可以是 PyTorch 官方优化器或者自定义的 sgd 函数)。

    Returns:
        包含 (本轮次平均训练损失, 本轮次训练准确率) 的元组。
    """
    # 1. 开启训练模式 (与从零实现无关，但是有助于兼容官方实现)
    if isinstance(net, nn.Module):
        net.train()

    # 2. 准备记账本: [总损失, 总正确数, 总样本数]
    metric = Accumulator(3)

    for X, y in train_iter:
        # 3. 计算前向传播结果并求损失
        y_hat = net(X)
        l = loss(y_hat, y)

        # 4. 反向传播与参数更新
        if isinstance(updater, torch.optim.Optimizer):
            # 使用 PyTorch 官方优化器的情况
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 如果用的是之前写的 sgd 函数
            l.sum().backward()
            updater(X.shape[0]) # 这里调用我们手写的 sgd(batch_size)

        # 5. 记录数据
        # l.sum() 是这一批次的总损失
        # count_correct(y_hat, y) 是这一批次对了几个
        # y.numel() 是这一批次总共有几个样本
        metric.add(float(l.sum()), count_correct(y_hat, y), y.numel())
    
    # 返回：平均损失 = 总损失 / 总样本数，准确率 = 总正确数 / 总样本数
    return metric[0] / metric[2], metric[1] / metric[2]

# 训练函数
def train_softmax(
    net: Callable[[Tensor], Tensor],
    train_iter: DataLoader,
    test_iter: DataLoader,
    loss: Callable[[Tensor, Tensor], Tensor],
    num_epochs: int,
    updater: Union[torch.optim.Optimizer, Callable]
) -> None:
    """训练并评估模型。
    
    Args:
        net: 神经网络模型。
        train_iter: 训练集数据加载器。
        test_iter: 测试集数据加载器。
        loss: 损失函数。
        num_epochs: 训练轮数。
        updater: 参数更新函数 (如 sgd)。
    """
    print(f"开始训练，总轮数: {num_epochs}。")

    for epoch in range(num_epochs):
        # 1. 执行一轮训练
        train_metrics = train_softmax_epoch(net, train_iter, loss, updater)

        # 2. 在测试集上评估准确率
        test_acc = evaluate_accuracy(net, test_iter)

        # 3. 打印当前轮次的进度
        train_loss, train_acc = train_metrics
        print(f"Epoch {epoch + 1}: "
              f"Loss = {train_loss:.4f}, "
              f"Train Acc = {train_acc:.4f}, "
              f"Test Acc = {test_acc:.4f}")
        
    print("训练完成！")

# ==============================================================================
# SECTION 5: 可视化与分析 (Visualization & Analysis)
# 作用：训练结束后的展示，通常依赖于前面的所有内容。
# ==============================================================================
def get_fashion_mnist_labels(labels: Tensor) -> list[str]:
    """返回 Fashion-MNIST 数据集的文本标签。
    
    Args:
        labels: 输入的标签张量。

    Returns:
        返回数字标签对应的字符串列表。
    """
    # 1. 定义一个列表，索引 0-9 严格对应官方定义的类别名称
    text_labels: list[str] = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]

    # 2. 列表推导式：遍历输入的 labels 张量
    # int(i) 将 Tensor 里的数字转为 Python 整数
    # text_labels[...] 根据这个整数从列表中取词
    return [text_labels[int(i)] for i in labels]

def show_images(
        imgs: Tensor,
        num_rows: int,
        num_cols: int,
        titles: Optional[list[str]] = None,
        scale: float = 1.5
) -> None:
    """绘制图像列表。
    
    Args:
        imgs: 输入图像集。
        num_rows: 图片行数。
        num_cols: 图片列数。
        titles: 输入图片对应的标签 (可选)。
        scale: 缩放比例。
    """
    # 1. 准备画布
    figsize = (num_cols * scale, num_rows * scale)
    # plt.subplots 会创建一个大图 (Figure) 和一堆小图区域 (Axes)
    # axes 是一个二维数组，比如 2 行 9 列
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # 2. 展平坐标轴
    # axes 原本是 [[ax1, ax2...], [ax10, ax11...]]
    # flatten() 把它变成一维列表 [ax1, ax2, ..., ax18]
    # 这样我们就可以用一个循环直接遍历所有的位置
    axes = axes.flatten()

    # 3. 循环画图
    # zip(axes, imgs) 把“格子”和“图片”一一对应地打包
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        # 绘制前，需将 Tensor 转换为 NumPy
        # 【核心点】img.squeeze().numpy()
        # img 的形状原本是 (1, 28, 28) -> (通道, 高, 宽)
        # squeeze() 会去掉大小为 1 的维度，变成 (28, 28)
        # numpy() 将 PyTorch 张量转为 NumPy 数组，因为 matplotlib 只认识 NumPy
        ax.imshow(img.squeeze().numpy(), cmap='gray')

        # 隐藏坐标轴上的刻度（0, 5, 10...这些数字），让图片看起来更整洁
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        # 如果传了标题列表，就给每个小图加上标题
        if titles:
            ax.set_title(titles[i])

    # 自动调整子图之间的间距，防止标题和图片重叠在一起
    plt.tight_layout()
    plt.show()

class DataStreamVisualizer:
    """一个用于顺序可视化预测结果的工程组件。"""

    def __init__(self, net: Callable[[Tensor], Tensor], data_iter: DataLoader):
        """
        Args:
            net: 已经训练好的模型。
            data_iter: 数据加载器。 
        """
        self.net = net
        self.data_iter = data_iter
        # 初始化迭代器
        self._stream: Iterator = iter(data_iter)

    def show_next(self, n: int = 6) -> None:
        """展示下一组的预测结果。"""
        # 1. 准备数据
        try:
            X, y = next(self._stream)
        except StopIteration:
            # 自动重置逻辑
            print(">>> 已到达数据集末尾，重置迭代器...")
            self.reset()
            X, y = next(self._stream)

        # 2. 准备真实标签的文字
        trues: list[str] = get_fashion_mnist_labels(y)

        # 3. 得到模型的预测结果
        # 经过 net(X) 得到概率，再用 argmax(axis=1) 得到最高概率的索引
        with torch.no_grad():
            if isinstance(self.net, nn.Module):
                self.net.eval() # 确保在测试模式
            preds: list[str] = get_fashion_mnist_labels(self.net(X).argmax(axis=1))

        # 4. 组合标题：格式为 "True: 实际 \n Pred:预测"
        # 如果预测错误，我们在标题前加 "X" 标记
        titles: list[str] = [
            f"{'√' if t == p else 'X'} {t}\n->{p}"
            for t, p in zip(trues, preds)
        ]

        # 5. 调用之前写好的 show_images 函数
        # X[:n] 取前 n 张图
        show_images(X[:n], 1, n, titles=titles[:n])
        print(f">>> 成功展示了 {n} 张样本")

    def reset(self) -> None:
        """手动重置流到数据集开头。"""
        self._stream = iter(self.data_iter)