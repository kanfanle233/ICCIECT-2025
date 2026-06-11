"""
散点图展示：二维数据点的分布与趋势。

教学重点：
  - 使用 matplotlib 绘制散点图（Scatter Plot）
  - 散点图常用于观察两个变量之间的相关性
  - 数据点的位置由 (x, y) 坐标对决定
"""

# --- 1. 导入与后端设置 ---
import matplotlib

matplotlib.use("Agg")  # 使用非交互式后端，适合脚本环境保存图片

import matplotlib.pyplot as plt

# --- 2. 数据准备 ---
x = [1, 2, 3, 4, 5]   # x 轴数据
y = [4, 7, 2, 9, 13]   # y 轴数据，与 x 一一对应

# --- 3. 绘制散点图 ---
plt.scatter(x, y)         # 绘制散点，每个点由 (x[i], y[i]) 定位
plt.title("Scatter Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.tight_layout()        # 自动调整布局，避免标签被裁剪
plt.savefig("q4_scatter.png", dpi=200)  # 保存为 PNG 图片，分辨率 200 dpi
