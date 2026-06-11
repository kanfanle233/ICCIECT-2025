"""
加载 MNIST 手写数字数据集，可视化样本图像及其标签。
教学重点：torchvision.datasets.MNIST 加载数据集、RandomAffine 数据增强、subplot 批量展示图像。
"""

import os
from torchvision import datasets as ds, transforms
import matplotlib.pyplot as ppl

# --- 1. 检测 MNIST 数据路径并加载数据集 ---
# 检测 MNIST 数据实际存放位置
if os.path.exists('../资源/MNIST/raw/'):
    root = '../资源/'
elif os.path.exists('../MNIST/raw/'):
    root = '../'
else:
    root = '../资源/'
# --- 2. 加载 MNIST 数据集并可视化 ---
tr = transforms.RandomAffine(45, (0.3,0.3))  # 数据增强：随机旋转45度、随机平移30%
dataset = ds.MNIST(root=root, train=True, download=True)  # 加载训练集
print(len(dataset))  # 打印样本总数（60000）
_, axes = ppl.subplots(5, 10)  # 创建 5行10列 的子图网格
for row in range(5):
    for col in range(10):
        n = row*10 + col
        img, label = dataset[n]
        img = tr(img)
        print(label, img.width, img.height)
        ax = axes[row][col]
        ax.imshow(img)
        ax.set_title(str(label))  # ppl.title()
        ax.axis("off")     # 不显示坐标轴
ppl.show()