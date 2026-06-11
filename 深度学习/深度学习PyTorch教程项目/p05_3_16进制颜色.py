"""
matplotlib 中使用 16 进制颜色码绘制多条曲线。
教学重点：16 进制颜色表示法（#RRGGBB）、多曲线叠加显示、图例 legend。
"""

from matplotlib import pyplot as ppl
import numpy as np

# --- 1. 生成数据 ---
x = np.linspace(-np.pi, np.pi, 500)  # -pi ~ pi
y1 = np.sin(x)
y2 = np.cos(x)

# 用16进制表示颜色
ppl.plot(x, y1+y2, label="sin+cos", color="#FF0000") # 红色
ppl.plot(x, y1-y2, label="sin-cos", color="#00FF00") # 绿色
ppl.plot(x, y1*y2, label="sin*cos", color="#0000FF") # 蓝色
ppl.plot(x, y1, label="sin", color="#B3B3B3") # 浅灰色
ppl.plot(x, y2, label="cos", color="#0A0A0A") # 深灰色
ppl.legend()
ppl.show()

