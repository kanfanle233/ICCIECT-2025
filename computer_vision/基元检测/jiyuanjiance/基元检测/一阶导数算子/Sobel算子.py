"""
教学示例：Sobel算子

- 功能：演示 一阶导数边缘检测 中与“Sobel算子”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""

import numpy as np


def sobel(image):
    """
    Sobel算子边缘检测

    参数:
        image: 输入灰度图像 (2D numpy数组)

    返回:
        magnitude: 梯度幅值（边缘强度）
    """
    h, w = image.shape
    magnitude = np.zeros((h - 2, w - 2))

    # Sobel算子核（加权平滑）
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])  # 水平方向（检测垂直边缘）

    kernel_y = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])  # 垂直方向（检测水平边缘）

    # 遍历图像（排除边界）
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # 提取3x3邻域
            neighborhood = image[i - 1:i + 2, j - 1:j + 2]

            # 计算梯度
            gx = np.sum(neighborhood * kernel_x)
            gy = np.sum(neighborhood * kernel_y)

            # 计算幅值
            magnitude[i - 1, j - 1] = np.sqrt(gx ** 2 + gy ** 2)

    return magnitude


# 使用卷积的快速版本
from scipy import signal


def sobel_fast(image):
    """使用卷积的快速Sobel实现"""
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gx = signal.convolve2d(image, kernel_x, mode='valid')
    gy = signal.convolve2d(image, kernel_y, mode='valid')

    return np.sqrt(gx ** 2 + gy ** 2)


# 返回梯度分量和方向的完整版本
def sobel_full(image):
    """返回完整的Sobel检测结果"""
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gx = signal.convolve2d(image, kernel_x, mode='valid')
    gy = signal.convolve2d(image, kernel_y, mode='valid')

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.arctan2(gy, gx) * 180 / np.pi

    return gx, gy, magnitude, direction


# 测试代码
if __name__ == "__main__":
    # 创建测试图像（包含水平边缘）
    img = np.array([
        [10, 10, 10, 10, 10],
        [10, 10, 10, 10, 10],
        [200, 200, 200, 200, 200],
        [200, 200, 200, 200, 200],
        [200, 200, 200, 200, 200]
    ], dtype=np.float64)

    print("输入图像:")
    print(img)

    # 测试不同版本
    edges1 = sobel(img)
    edges2 = sobel_fast(img)
    gx, gy, mag, direc = sobel_full(img)

    print("\nSobel边缘检测结果（幅值）:")
    print(edges1)

    print("\n梯度分量 Gx（水平变化）:")
    print(gx)

    print("\n梯度分量 Gy（垂直变化）:")
    print(gy)

    print("\n梯度方向（度）:")
    print(direc)