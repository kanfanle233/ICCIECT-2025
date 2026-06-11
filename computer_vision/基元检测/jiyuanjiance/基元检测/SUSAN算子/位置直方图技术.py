# -*- coding: utf-8 -*-
"""
教学示例：位置直方图技术

- 功能：演示 基元检测 中与“位置直方图技术”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""

import cv2
import numpy as np


def position_histogram_1d(image, bins=32):
    """
    一维位置直方图
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape

    # 计算X方向直方图
    hist_x = np.zeros(bins)
    hist_y = np.zeros(bins)

    for i in range(h):
        for j in range(w):
            x_idx = int(j * bins / w)
            y_idx = int(i * bins / h)
            hist_x[x_idx] += gray[i, j]
            hist_y[y_idx] += gray[i, j]

    # 归一化
    if np.sum(hist_x) > 0:
        hist_x = hist_x / np.sum(hist_x)
    if np.sum(hist_y) > 0:
        hist_y = hist_y / np.sum(hist_y)

    return hist_x, hist_y


def position_histogram_2d(image, bins=(16, 16)):
    """
    二维位置直方图
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape
    bins_y, bins_x = bins

    hist_2d = np.zeros((bins_y, bins_x))

    for i in range(h):
        for j in range(w):
            x_idx = int(j * bins_x / w)
            y_idx = int(i * bins_y / h)
            hist_2d[y_idx, x_idx] += gray[i, j]

    if np.sum(hist_2d) > 0:
        hist_2d = hist_2d / np.sum(hist_2d)

    return hist_2d


def draw_histogram_1d(hist, title="直方图", width=400, height=300):
    """
    绘制一维直方图（使用OpenCV）
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    if len(hist) == 0 or np.max(hist) == 0:
        return img

    # 归一化到图像高度
    hist_norm = hist / np.max(hist) * (height - 50)

    bar_width = width // len(hist)

    for i, val in enumerate(hist_norm):
        x1 = i * bar_width
        x2 = (i + 1) * bar_width - 2
        y1 = height - 30
        y2 = int(height - 30 - val)

        cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 255), -1)

    cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return img


def draw_histogram_2d(hist, title="二维直方图", size=300):
    """
    绘制二维直方图（使用OpenCV）
    """
    h, w = hist.shape
    img = np.zeros((size, size, 3), dtype=np.uint8)

    if np.max(hist) > 0:
        hist_norm = hist / np.max(hist) * 255
    else:
        hist_norm = hist

    # 缩放直方图到图像大小
    for i in range(h):
        for j in range(w):
            y1 = int(i * size / h)
            y2 = int((i + 1) * size / h)
            x1 = int(j * size / w)
            x2 = int((j + 1) * size / w)

            val = int(hist_norm[i, j])
            color = (0, 0, val)  # BGR格式，蓝色通道表示强度
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def create_test_image():
    """创建测试图像"""
    img = np.zeros((300, 300, 3), dtype=np.uint8)

    # 添加一个矩形
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)

    # 添加一个圆形
    cv2.circle(img, (200, 200), 50, (200, 200, 200), -1)

    # 添加一条直线
    cv2.line(img, (50, 250), (250, 250), (150, 150, 150), 5)

    return img


def main():
    """主函数"""

    # 创建测试图像
    print("创建测试图像...")
    img = create_test_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. 一维位置直方图
    hist_x, hist_y = position_histogram_1d(img)

    # 2. 二维位置直方图
    hist_2d = position_histogram_2d(img, bins=(16, 16))

    # 绘制直方图
    hist_x_img = draw_histogram_1d(hist_x, "X方向位置直方图")
    hist_y_img = draw_histogram_1d(hist_y, "Y方向位置直方图")
    hist_2d_img = draw_histogram_2d(hist_2d, "二维位置直方图")

    # 显示结果
    cv2.imshow('1. 原始图像', img)
    cv2.imshow('2. 灰度图像', gray)
    cv2.imshow('3. X方向直方图', hist_x_img)
    cv2.imshow('4. Y方向直方图', hist_y_img)
    cv2.imshow('5. 二维直方图', hist_2d_img)

    print("\n直方图统计:")
    print(f"X方向直方图: 最小值={np.min(hist_x):.4f}, 最大值={np.max(hist_x):.4f}")
    print(f"Y方向直方图: 最小值={np.min(hist_y):.4f}, 最大值={np.max(hist_y):.4f}")
    print(f"二维直方图形状: {hist_2d.shape}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 更简洁的版本
def simple_position_histogram_demo(image_path=None):
    """
    最简版本的位置直方图演示
    """
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            print("使用测试图像")
            img = create_test_image()
    else:
        img = create_test_image()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 计算位置直方图
    bins = 32
    hist_x = np.zeros(bins)
    hist_y = np.zeros(bins)

    for i in range(h):
        for j in range(w):
            x_idx = int(j * bins / w)
            y_idx = int(i * bins / h)
            hist_x[x_idx] += gray[i, j]
            hist_y[y_idx] += gray[i, j]

    # 归一化
    hist_x = hist_x / np.sum(hist_x)
    hist_y = hist_y / np.sum(hist_y)

    # 简单打印结果
    print("=" * 50)
    print("位置直方图结果")
    print("=" * 50)
    print(f"X方向直方图 (前10个值): {hist_x[:10]}")
    print(f"Y方向直方图 (前10个值): {hist_y[:10]}")
    print(f"X方向最大值位置: {np.argmax(hist_x)}")
    print(f"Y方向最大值位置: {np.argmax(hist_y)}")

    return hist_x, hist_y


if __name__ == "__main__":
    # 运行主程序
    main()

    # 或运行简单版本
    # simple_position_histogram_demo()