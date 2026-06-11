"""
二元线性分类示例。

教学重点：用一条边界把两类样本分开。
"""

# 导入所需的库
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# 设置中文和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建一个2x2的子图布局，figsize设置图形大小
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 生成并绘制不同噪声的数据集
# 注意：每个子图需要使用不同的数据集和标题
x1, y1 = make_moons(n_samples=100, noise=0.0, random_state=42)
axes[0, 0].scatter(x1[:, 0], x1[:, 1], c=y1, cmap=plt.cm.RdBu)
axes[0, 0].set_title('无噪声 (noise=0.0)')

x2, y2 = make_moons(n_samples=100, noise=0.1, random_state=42)
axes[0, 1].scatter(x2[:, 0], x2[:, 1], c=y2, cmap=plt.cm.RdBu)
axes[0, 1].set_title('少量噪声 (noise=0.1)')

x3, y3 = make_moons(n_samples=100, noise=0.2, random_state=42)
axes[1, 0].scatter(x3[:, 0], x3[:, 1], c=y3, cmap=plt.cm.RdBu)
axes[1, 0].set_title('中等噪声 (noise=0.2)')

# 这是一个更复杂的例子：数据不平衡
# 这里使用 n_samples 来控制不同类别的数量
x4, y4 = make_moons(n_samples=(80, 10), noise=0.2, random_state=42)
axes[1, 1].scatter(x4[:, 0], x4[:, 1], c=y4, cmap=plt.cm.RdBu)
axes[1, 1].set_title('不平衡数据 (80:10)')

# 自动调整子图参数，使之填充整个图像区域
plt.tight_layout()

# 显示图形
plt.show()