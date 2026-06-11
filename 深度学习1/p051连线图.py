"""
Matplotlib 连线图示例。

教学重点：用数组保存点坐标，再把点连成线。
"""

from matplotlib import pyplot as ppl
import numpy as np

# --- 基础代码保持不变 ---
# x 轴数据是 -pi 到 pi 之间的 500 个随机点
x = np.linspace(-np.pi,np.pi, num=500)
print(x)
print(x.shape)


# 计算 sin 和 cos 的 y 值
y1 = np.sin(x)
y2 = np.cos(x)




# --- 绘图部分 ---

# 绘制 sin (暗红色)
ppl.scatter(x, y1, label="sin", color="darkred")

# 绘制 cos (暗蓝色)
ppl.scatter(x, y2, label="cos", color="darkblue")


# (推荐) 设置Y轴的范围，与上面的过滤值保持一致


# 添加图例和网格，显示图表
ppl.legend()
ppl.grid(True)
ppl.show()