"""
Matplotlib 十六进制颜色示例。

教学重点：颜色字符串如何控制图形显示。
十六进制颜色格式为 '#RRGGBB'，RR/GG/BB 各占两个十六进制位，范围 00~FF。
"""

from matplotlib import pyplot as ppl
import numpy as np

# --- 1. 数据准备 ---
x = np.linspace(-np.pi, np.pi, num=500)

y1 = np.sin(x)  # 正弦值
y2 = np.cos(x)  # 余弦值

# --- 2. 绘图（使用十六进制颜色代码） ---
ppl.plot(x, y1 + y2, label='sin+cos', color='#FF0000')   # 红色
ppl.plot(x, y1 - y2, label='sin-cos', color='#008000')   # 绿色
ppl.plot(x, y1 * y2, label='sin*cos', color='#0000FF')   # 蓝色
ppl.plot(x, y1, label='sin', color='#00FFFF')             # 青色
ppl.plot(x, y2, label='cos', color='#FF00FF')             # 品红色

# --- 3. 显示图例和图形 ---
ppl.legend()
ppl.show()