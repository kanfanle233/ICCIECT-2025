"""
matplotlib 子图绘制示例，将 sin 与 cos 显示在同一行的两个子图中。
教学重点：subplot 创建子图、plot 绘制折线图、图例和标题设置。
"""

from matplotlib import pyplot as ppl
import numpy as np

# --- 1. 生成数据 ---
x = np.linspace(-np.pi, np.pi, 500)  # 在 -pi ~ pi 区间均匀取 500 个点
y1 = np.sin(x)
y2 = np.cos(x)

# 1行2列，第1个子图
ppl.subplot(1, 2, 1)
ppl.plot(x, y1, label="sin", color="red")
ppl.legend()
ppl.title("Sin")

# 1行2列，第2个子图
ppl.subplot(1, 2, 2)
ppl.plot(x, y2, label="cos", color="blue")
ppl.legend()
ppl.title("Cos")

ppl.show()

