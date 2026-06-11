"""
教学示例：Prewitti与Sobel对比

- 功能：演示 一阶导数边缘检测 中与“Prewitti与Sobel对比”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""

import numpy as np


def safe_comparison():
    """安全的Prewitt和Sobel比较"""

    print("=" * 50)
    print("Prewitt vs Sobel 安全比较")
    print("=" * 50)

    # 创建测试图像
    img = np.array([
        [10, 10, 200, 200, 200],
        [10, 10, 200, 200, 200],
        [10, 10, 200, 200, 200],
        [10, 10, 200, 200, 200],
        [10, 10, 200, 200, 200]
    ], dtype=np.float64)

    print("\n测试图像（垂直边缘）:")
    print(img)

    # 定义核
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    from scipy import signal

    # 计算响应
    pre_gx = signal.convolve2d(img, prewitt_x, mode='valid')
    sob_gx = signal.convolve2d(img, sobel_x, mode='valid')

    pre_max = np.max(pre_gx)
    sob_max = np.max(sob_gx)

    print(f"\nPrewitt最大响应: {pre_max}")
    print(f"Sobel最大响应:   {sob_max}")

    # 安全计算比值
    if pre_max != 0:
        if not np.isnan(pre_max) and not np.isnan(sob_max):
            ratio = sob_max / pre_max
            print(f"\nSobel/Prewitt响应比: {ratio:.2f}")
            print(f"结论: Sobel响应是Prewitt的{ratio:.2f}倍")
        else:
            print("\n存在NaN值，无法计算")
    else:
        print("\nPrewitt响应为0，无法计算比值")
        print("可能原因: 图像区域没有边缘或核用错了方向")


# 运行安全比较
safe_comparison()



#mode='valid'和same的区别
# == == = 生活例子 == == =
#
# 想象你用3cm×3
# cm的印章在5cm×5
# cm的纸上盖章：
#
# mode = 'valid'（有效模式）：
# - 只印印章完全在纸上的位置
# - 印章不能超出纸边
# - 结果：3×3
# 个印痕
#
# mode = 'same'（相同模式）：
# - 印痕中心和每个格子对齐
# - 印章可以部分超出纸边
# - 结果：5×5
# 个印痕
# """)

import numpy as np
def size_comparison():
    """尺寸对比"""

    # 输入图像
    img = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ])

    # 3x3卷积核
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])

    from scipy import signal

    # valid模式
    valid_result = signal.convolve2d(img, kernel, mode='valid')

    # same模式
    same_result = signal.convolve2d(img, kernel, mode='same')

    print("输入图像尺寸:", img.shape)  # (5, 5)
    print("valid输出尺寸:", valid_result.shape)  # (3, 3)
    print("same输出尺寸: ", same_result.shape)  # (5, 5)