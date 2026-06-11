"""
教学示例：Prewitt算子

- 功能：演示 一阶导数边缘检测 中与“Prewitt算子”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""

import numpy as np
from scipy import signal


def prewitt_fast(image):
    """
    使用卷积的快速Prewitt实现
    """
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    gx = signal.convolve2d(image, kernel_x, mode='valid')
    gy = signal.convolve2d(image, kernel_y, mode='valid')

    return np.sqrt(gx ** 2 + gy ** 2)


# 测试
img = np.array([
    [10, 10, 10, 10, 10],
    [10, 10, 10, 10, 10],
    [200, 200, 200, 200, 200],
    [200, 200, 200, 200, 200],
    [200, 200, 200, 200, 200]
], dtype=np.float64)

edges = prewitt_fast(img)
print("Prewitt边缘检测结果:")
print(edges)