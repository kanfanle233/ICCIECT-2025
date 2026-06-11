"""
Matplotlib 基础折线图示例。

教学重点：横轴、纵轴和曲线之间的关系。
"""

from matplotlib import pyplot as ppl
import numpy as np

# --- 基础代码保持不变 ---
# x 轴数据是 -pi 到 pi 之间的 500 个随机点
x = (np.random.rand(500) * 2 - 1) * np.pi

# 计算 sin 和 cos 的 y 值
y1 = np.sin(x)
y2 = np.cos(x)

# --- 新增代码部分 ---
# 计算 tan 和 cot 的 y 值
y3 = np.tan(x)
y4 = 1 / np.tan(x)  # cot(x) = 1 / tan(x)

# (关键步骤) 过滤掉 tan 和 cot 的极端值，避免图像失真
# 我们设定一个Y轴的显示范围，比如 -5 到 5
y_limit = 5
y3[np.abs(y3) > y_limit] = np.nan  # 将超出范围的点设为 NaN (Not a Number)
y4[np.abs(y4) > y_limit] = np.nan  # Matplotlib 在绘图时会自动忽略 NaN 的点

# --- 绘图部分 ---

# 绘制 sin (暗红色)
ppl.scatter(x, y1, label="sin", color="darkred")

# 绘制 cos (暗蓝色)
ppl.scatter(x, y2, label="cos", color="darkblue")

# (新增) 绘制 tan (暗绿色)
ppl.scatter(x, y3, label="tan", color="darkgreen")

# (新增) 绘制 cot (暗紫色)
ppl.scatter(x, y4, label="cot", color="darkmagenta")

# (推荐) 设置Y轴的范围，与上面的过滤值保持一致
ppl.ylim(-y_limit, y_limit)

# 添加图例和网格，显示图表
ppl.legend()
ppl.grid(True)
ppl.show()