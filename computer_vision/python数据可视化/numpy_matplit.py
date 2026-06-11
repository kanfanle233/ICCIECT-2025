"""
教学示例：numpy matplit

- 功能：演示 数据可视化 中与“numpy matplit”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""

import matplotlib.pyplot as plt
import numpy as np

# 解决 matplotlib 在 macOS / PyCharm 中中文显示为方块或警告 Glyph missing 的问题
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

x = np.arange(10)  # 取值依次为 0-9 的等差数列
y = np.sin(x)
z = np.cos(x)

plt.plot(x, y, marker="*", linewidth=3, linestyle="--", color="red")  # marker 设置数据点标记
plt.plot(x, z)
plt.title("matplotlib 正弦余弦曲线")
plt.xlabel("x轴")
plt.ylabel("y轴")
plt.legend(["Y轴", "Z"], loc="upper right")  # 设置图例
plt.grid(True)
plt.show()
