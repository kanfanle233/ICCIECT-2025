"""
matplotlib 绑定图基础示例，展示 sin 与 cos 散点图的绘制。
教学重点：pyplot 散点图 scatter、图例 legend、坐标范围设置。
"""

from matplotlib import pyplot as ppl
import numpy as np

# --- 1. 生成数据 ---
x = (np.random.rand(500) * 2 - 1) * np.pi  # -pi ~ pi 的随机横坐标
y1 = np.sin(x)
y2 = np.cos(x)

ppl.scatter(x, y1, label="sin", color="red")
ppl.scatter(x, y2, label="cos", color="blue")
ppl.legend()
ppl.show()

