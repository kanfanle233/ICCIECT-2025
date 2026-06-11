"""
可视化 make_moons 数据集在不同噪声水平下的分布，理解二分类任务的样本形态。
教学重点：sklearn make_moons 生成月牙形二分类数据、subplot 多子图对比、散点图着色。
"""

import matplotlib.pyplot as ppl
from sklearn.datasets import make_moons

ppl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
ppl.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# --- 1. 生成不同噪声水平下的月牙形数据集 ---
# ppl.subplot(rows, cols, order)
_, axes = ppl.subplots(2, 2, figsize=(12, 10))

# 1. 默认参数（无噪声）
X1, y1 = make_moons(n_samples=100, noise=0.0, random_state=42)
axes[0,0].scatter(X1[:,0], X1[:,1], c=y1, cmap=ppl.cm.RdBu)  # 按类别着色
axes[0,0].set_title('无噪声 (noise=0.0)')

# 2. 添加少量噪声
X2, y2 = make_moons(n_samples=100, noise=0.1, random_state=42)
axes[0,1].scatter(X2[:,0], X2[:,1], c=y2, cmap=ppl.cm.RdBu)
axes[0,1].set_title('少量噪声 (noise=0.1)')

# 3. 添加中等噪声
X3, y3 = make_moons(n_samples=100, noise=0.2, random_state=42)
axes[1,0].scatter(X3[:,0], X3[:,1], c=y3, cmap=ppl.cm.RdBu)
axes[1,0].set_title('中等噪声 (noise=0.2)')

# 4. 不平衡数据集（80:20 比例）
X4, y4 = make_moons(n_samples=(80, 20), noise=0.1, random_state=42)
axes[1,1].scatter(X4[:,0], X4[:,1], c=y4, cmap=ppl.cm.RdBu)
axes[1,1].set_title('不平衡数据集 (80:20)')

ppl.tight_layout()  # 自动调整子图间距，避免重叠
ppl.show()
