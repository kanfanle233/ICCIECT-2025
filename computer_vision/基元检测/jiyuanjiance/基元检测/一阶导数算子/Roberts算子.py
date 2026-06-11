"""
教学示例：Roberts算子

- 功能：演示 一阶导数边缘检测 中与“Roberts算子”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""

import numpy as np


def roberts_operator_complete():
    """完整的Robert算子实现"""

    print("=== 完整Robert算子实现 ===")

    # 模拟一个简单的测试图像
    # 创建一个4x4的numpy数组，模拟灰度图像
    # 数值范围0-255，代表灰度值
    image = np.array([
        [100, 100, 100, 100],  # 第一行：全100（较暗区域）
        [100, 150, 200, 100],  # 第二行：从左到右先变亮再变暗
        [100, 200, 250, 100],  # 第三行：中间最亮(250)
        [100, 100, 100, 100]  # 第四行：全100（较暗区域）
    ])

    print("输入图像：")
    print(image)

    # 获取图像的高度(h)和宽度(w)
    h, w = image.shape

    # 初始化结果矩阵
    # 因为使用2x2窗口，结果矩阵尺寸比原图小1
    result_45 = np.zeros((h - 1, w - 1))  # 存储45°方向响应
    result_135 = np.zeros((h - 1, w - 1))  # 存储135°方向响应
    magnitude = np.zeros((h - 1, w - 1))  # 存储梯度幅值

    # 遍历每个2×2窗口
    # i循环：遍历每一行（直到倒数第二行）
    for i in range(h - 1):
        # j循环：遍历每一列（直到倒数第二列）
        for j in range(w - 1):
            # 提取当前2×2窗口
            # image[i:i+2, j:j+2] 表示从(i,j)开始的2x2区域
            window = image[i:i + 2, j:j + 2]

            # 将2×2窗口的四个像素分别命名为a,b,c,d
            # a: 左上角像素  b: 右上角像素
            # c: 左下角像素  d: 右下角像素
            a, b, c, d = window[0, 0], window[0, 1], window[1, 0], window[1, 1]

            # 计算Robert算子的两个响应
            # G45: 45°方向响应 (右下减左上)
            # 正值表示沿45°方向（↘）灰度增加
            # 负值表示沿45°方向灰度减少
            G45 = d - a

            # G135: 135°方向响应 (右上减左下)
            # 正值表示沿135°方向（↙）灰度增加
            # 负值表示沿135°方向灰度减少
            G135 = b - c

            # 存储计算结果
            result_45[i, j] = G45  # 保存45°方向响应
            result_135[i, j] = G135  # 保存135°方向响应

            # 计算梯度幅值（边缘强度）
            # 使用勾股定理：幅值 = √(G45² + G135²)
            # 幅值越大，表示该位置的边缘越明显
            magnitude[i, j] = np.sqrt(G45 ** 2 + G135 ** 2)

    # 输出结果
    print("\n45°方向响应 (d-a)：")
    print(result_45)

    print("\n135°方向响应 (b-c)：")
    print(result_135)

    print("\n梯度幅值（边缘强度）：")
    print(magnitude)

    # 返回计算结果，方便后续使用
    return result_45, result_135, magnitude


# 调用函数执行Robert算子
roberts_operator_complete()