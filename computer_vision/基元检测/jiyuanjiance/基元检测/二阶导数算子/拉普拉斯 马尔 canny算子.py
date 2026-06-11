# -*- coding: utf-8 -*-
"""
教学示例：拉普拉斯 马尔 canny算子

- 功能：演示 二阶导数与边缘检测 中与“拉普拉斯 马尔 canny算子”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""
from pathlib import Path

import cv2
import numpy as np



BASE_DIR = Path(__file__).resolve().parent

def local_path(name: str) -> str:
    """返回脚本目录下的资源路径，避免依赖当前工作目录。"""
    return str(BASE_DIR / name)
def laplacian_manual(image):
    """
    手动实现拉普拉斯算子（使用4邻域模板）

    模板:
    [0  -1   0]
    [-1  4  -1]
    [0  -1   0]
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray = image.astype(np.float64)

    h, w = gray.shape
    result = np.zeros((h - 2, w - 2), dtype=np.float64)

    # 拉普拉斯模板（你给的矩阵）
    laplacian_kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.float64)

    print("=" * 50)
    print("拉普拉斯算子实现")
    print("=" * 50)
    print("\n使用的拉普拉斯模板:")
    print(laplacian_kernel)
    print(f"模板和: {np.sum(laplacian_kernel)} (应为0)")
    print(f"模板形状: {laplacian_kernel.shape}")

    # 遍历图像
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # 提取3x3邻域
            neighborhood = gray[i - 1:i + 2, j - 1:j + 2]

            # 卷积计算
            result[i - 1, j - 1] = np.sum(neighborhood * laplacian_kernel)

    print(f"\n输入图像尺寸: {gray.shape}")
    print(f"输出结果尺寸: {result.shape}")

    return result


def normalize_for_display(image):
    """归一化图像用于显示"""
    abs_img = np.abs(image)
    norm_img = cv2.normalize(abs_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return norm_img


# 测试
if __name__ == "__main__":
    # 1. 创建或读取图像
    img = cv2.imread(local_path("lena.png"))
    if img is None:
        print("\n创建测试图像...")
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        # 添加矩形
        cv2.rectangle(img, (50, 50), (250, 250), (255, 255, 255), 2)
        # 添加圆形
        cv2.circle(img, (150, 150), 80, (255, 255, 255), 2)
        # 添加直线
        cv2.line(img, (50, 150), (250, 150), (255, 255, 255), 2)
        # 添加对角线
        cv2.line(img, (50, 250), (250, 50), (255, 255, 255), 2)

    # 2. 应用拉普拉斯算子
    result = laplacian_manual(img)

    # 3. 处理结果用于显示
    result_positive = np.maximum(result, 0)  # 正值（从暗到亮）
    result_negative = np.maximum(-result, 0)  # 负值（从亮到暗）
    result_norm = normalize_for_display(result)

    # 4. 过零点检测
    zero_cross = np.zeros_like(result_norm)
    h, w = result.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if (result[i, j] * result[i - 1, j] < 0) or \
                    (result[i, j] * result[i + 1, j] < 0) or \
                    (result[i, j] * result[i, j - 1] < 0) or \
                    (result[i, j] * result[i, j + 1] < 0):
                zero_cross[i, j] = 255

    # 5. 调整图像大小以便显示
    img_resized = cv2.resize(img, (result_norm.shape[1], result_norm.shape[0]))

    # 6. 创建彩色映射结果
    result_color = cv2.applyColorMap(result_norm, cv2.COLORMAP_JET)

    # 7. 显示结果
    print("\n显示结果...")

    cv2.imshow('1. 原始图像', img_resized)
    cv2.imshow('2. 拉普拉斯响应 (归一化)', result_norm)
    cv2.imshow('3. 拉普拉斯响应 (彩色)', result_color)
    cv2.imshow('4. 正值响应 (从暗到亮)', normalize_for_display(result_positive))
    cv2.imshow('5. 负值响应 (从亮到暗)', normalize_for_display(result_negative))
    cv2.imshow('6. 过零点检测', zero_cross)

    # 8. 显示模板信息
    info_img = np.zeros((200, 400, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(info_img, '拉普拉斯模板 (4邻域):', (10, 40), font, 0.7, (255, 255, 255), 2)
    cv2.putText(info_img, '[0  -1   0]', (10, 80), font, 0.7, (255, 255, 255), 2)
    cv2.putText(info_img, '[-1  4  -1]', (10, 120), font, 0.7, (255, 255, 255), 2)
    cv2.putText(info_img, '[0  -1   0]', (10, 160), font, 0.7, (255, 255, 255), 2)
    cv2.putText(info_img, f'模板和: 0', (10, 190), font, 0.5, (0, 255, 0), 1)

    cv2.imshow('7. 模板信息', info_img)

    print("\n按任意键关闭窗口...")
    print("提示: 点击任意图像窗口后按键盘任意键退出")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # 9. 保存结果
    # print("\n保存结果...")
    # cv2.imwrite('01_original.jpg', img_resized)
    # cv2.imwrite('02_laplacian_norm.jpg', result_norm)
    # cv2.imwrite('03_laplacian_color.jpg', result_color)
    # cv2.imwrite('04_positive.jpg', normalize_for_display(result_positive))
    # cv2.imwrite('05_negative.jpg', normalize_for_display(result_negative))
    # cv2.imwrite('06_zero_cross.jpg', zero_cross)
    # cv2.imwrite('07_template_info.jpg', info_img)

