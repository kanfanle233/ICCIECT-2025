"""
二元非线性分类可视化示例。

教学重点：非线性模型可以学习弯曲的分类边界。
make_moons 生成两个交错的半月形数据，是二分类的经典测试集。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --- 1. 设置随机种子以确保结果可重现 ---
torch.manual_seed(42)
np.random.seed(42)

# --- 2. 创建合成数据集 ---
# make_moons 生成两个交错的半月形，noise 控制噪声程度
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

# 将数据转换为 PyTorch 张量
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()


# --- 3. 决策边界可视化函数 ---
def plot_decision_boundary(model, X, y):
    """在二维平面上绘制模型的分类决策边界。"""
    # 生成网格点，覆盖数据所在区域
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 对每个网格点预测类别
    Z = model(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float())
    Z = (Z >= 0.5).float().numpy().reshape(xx.shape)

    # 用颜色填充决策区域，用散点标记原始数据
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title("决策边界")
    plt.show()


# 绘制决策边界
plot_decision_boundary(model, X.numpy(), y.numpy())


# 重复定义（保留原有结构）
def plot_decision_boundary(model, X, y):
    # ... (code for plotting decision boundary)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title("决策边界")
    plt.show()


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# --- 4. 绘制决策边界和损失曲线 ---
plot_decision_boundary(model, X.numpy(), y.numpy())

# 绘制损失曲线
plt.plot(losses)
plt.title('训练损失曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# --- 5. 可视化决策边界（完整版） ---
def plot_decision_boundary(model, X, y):
    """绘制完整的决策边界图，包含网格预测和数据散点。"""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float())
    Z = (Z >= 0.5).float().numpy().reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title("决策边界")
    plt.show()
