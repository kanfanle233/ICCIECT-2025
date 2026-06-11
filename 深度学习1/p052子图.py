"""
Matplotlib 子图示例。

教学重点：一个窗口中同时展示多个图表。
subplot(行数, 列数, 当前第几个) 可以在一张画布上并排显示多张图。
"""

from matplotlib import pyplot as ppl
import numpy as np

# --- 1. 数据准备 ---
# x 轴数据是 -pi 到 pi 之间的 500 个等间距点
x = np.linspace(-np.pi, np.pi, num=500)

y1 = np.sin(x)  # 正弦值
y2 = np.cos(x)  # 余弦值

# --- 2. 绘图部分 ---
# subplot(1,2,1) 表示：1 行 2 列的布局，当前画第 1 个子图（左边）
ppl.subplot(1, 2, 1)
ppl.plot(x, y1, label='sin', color="red")
ppl.legend()
ppl.title("Sin")

# subplot(1,2,2) 表示：1 行 2 列的布局，当前画第 2 个子图（右边）
ppl.subplot(1, 2, 2)
ppl.plot(x, y2, label='cos', color="blue")
ppl.legend()
ppl.title("Cos")

ppl.show()