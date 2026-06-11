"""
CIFAR-10 原始样本和标签查看脚本。

教学重点：
- CIFAR-10 图片原始形状常见为 (N, C, H, W)。
- N 是样本数量，C 是颜色通道 RGB=3，H/W 是图片高宽 32x32。
- labels 是整数类别编号，需要用 label_names 转成人能读懂的类别名。
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from runtime_compat import cifar10_root

# 读取一个批次
def load_batch(filename):
    """读取 CIFAR-10 官方二进制 batch，并整理成 PyTorch 常见的通道优先格式。"""
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']
        labels = batch[b'labels']
        data = data.reshape(-1, 3, 32, 32)  # 转成 (N, C, H, W)
        return data, labels

# CIFAR-10 标签名称
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 加载 data_batch_1（优先本地二进制；不存在时回退到 torchvision）
batch_file = Path(cifar10_root()) / "cifar-10-batches-py" / "data_batch_1"
if batch_file.exists():
    data, labels = load_batch(str(batch_file))
else:
    from torchvision import datasets
    try:
        ds = datasets.CIFAR10(root=cifar10_root(), train=True, download=True)
        data = ds.data.transpose(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        labels = ds.targets
    except Exception as e:
        raise RuntimeError(
            "未找到本地 CIFAR-10（cifar-10-batches-py/data_batch_1），且在线下载失败。"
            "请先准备好本地数据集后再运行。"
        ) from e

print("数据形状:", data.shape)   # (10000, 3, 32, 32)
print("标签数量:", len(labels))

# 随机展示几张图片
for i in range(5):
    img = data[i].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    plt.imshow(img)
    plt.title(label_names[labels[i]])
    plt.show()
