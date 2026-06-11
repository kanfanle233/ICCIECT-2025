# -*- coding: utf-8 -*-
"""
教学示例：Harris交叉点-T型交点

- 功能：演示 基元检测 中与“Harris交叉点-T型交点”相关的核心流程。
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
def detect_junctions(image_path):
    """
    检测图像中的交叉点(X型)和T型交点

    参数:
        image_path: 图像路径
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        # 创建测试图像
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        # 创建交叉点 (X型)
        cv2.line(img, (100, 100), (200, 200), (255, 255, 255), 2)
        cv2.line(img, (200, 100), (100, 200), (255, 255, 255), 2)
        # 创建T型交点
        cv2.line(img, (300, 100), (300, 200), (255, 255, 255), 2)
        cv2.line(img, (250, 150), (350, 150), (255, 255, 255), 2)
        # 创建普通角点 (L型)
        cv2.rectangle(img, (50, 250), (150, 350), (255, 255, 255), 2)

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Harris角点检测
    gray_float = np.float32(gray)
    harris_response = cv2.cornerHarris(gray_float, 2, 3, 0.04)
    harris_response = cv2.dilate(harris_response, None)

    # 获取角点坐标
    threshold = 0.01
    corner_coords = np.argwhere(harris_response > threshold * harris_response.max())

    # 2. 分析每个角点的局部区域，判断类型
    cross_points = []  # 交叉点 (X型)
    t_junctions = []  # T型交点
    l_corners = []  # 普通角点 (L型)

    # 窗口大小
    window_size = 15
    half = window_size // 2

    for y, x in corner_coords:
        # 提取局部区域
        y1, y2 = max(0, y - half), min(gray.shape[0], y + half + 1)
        x1, x2 = max(0, x - half), min(gray.shape[1], x + half + 1)
        patch = gray[y1:y2, x1:x2]

        if patch.size == 0:
            continue

        # 计算局部区域的梯度方向直方图
        sobelx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)

        # 计算梯度幅值和方向
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        direction = np.arctan2(sobely, sobelx) * 180 / np.pi

        # 只考虑梯度幅值较大的像素
        mag_threshold = np.mean(magnitude) * 0.5
        strong_pixels = magnitude > mag_threshold

        if np.sum(strong_pixels) < 10:
            continue

        # 计算梯度方向直方图
        hist, bins = np.histogram(direction[strong_pixels], bins=36, range=(-180, 180))

        # 平滑直方图
        hist = np.convolve(hist, [1, 2, 1], mode='same')

        # 找到峰值
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > np.mean(hist) * 1.2:
                peaks.append(i)

        # 根据峰值数量判断类型
        if len(peaks) >= 4:
            cross_points.append((x, y))
        elif len(peaks) == 3:
            t_junctions.append((x, y))
        elif len(peaks) == 2:
            l_corners.append((x, y))

    # 3. 可视化结果
    result = original.copy()

    # 绘制不同类型的点
    for x, y in l_corners:
        cv2.circle(result, (x, y), 5, (255, 0, 0), -1)  # 蓝色: L型角点
        cv2.putText(result, 'L', (x - 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    for x, y in t_junctions:
        cv2.circle(result, (x, y), 7, (0, 255, 0), -1)  # 绿色: T型交点
        cv2.putText(result, 'T', (x - 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    for x, y in cross_points:
        cv2.circle(result, (x, y), 7, (0, 0, 255), -1)  # 红色: 交叉点
        cv2.putText(result, 'X', (x - 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 显示结果
    cv2.imshow('Original', original)
    cv2.imshow('Junction Detection', result)

    # 显示Harris响应图
    harris_norm = cv2.normalize(harris_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    harris_color = cv2.applyColorMap(harris_norm, cv2.COLORMAP_HOT)
    cv2.imshow('Harris Response', harris_color)

    print(f"检测结果:")
    print(f"L型角点: {len(l_corners)}")
    print(f"T型交点: {len(t_junctions)}")
    print(f"交叉点: {len(cross_points)}")
    print(f"总计: {len(l_corners) + len(t_junctions) + len(cross_points)}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #
    # # 保存结果
    # cv2.imwrite('junction_detection.jpg', result)
    return l_corners, t_junctions, cross_points


# 更简洁的版本：使用OpenCV的cornerSubPix和角点分类
def detect_junctions_simple(image_path):
    """简化的交点检测"""
    img = cv2.imread(image_path)
    if img is None:
        # 创建测试图像
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        # 交叉点
        cv2.line(img, (100, 100), (200, 200), (255, 255, 255), 2)
        cv2.line(img, (200, 100), (100, 200), (255, 255, 255), 2)
        # T型点
        cv2.line(img, (300, 100), (300, 200), (255, 255, 255), 2)
        cv2.line(img, (250, 150), (350, 150), (255, 255, 255), 2)

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Harris角点检测
    dst = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    dst = cv2.dilate(dst, None)

    # 获取角点坐标
    _, dst_norm = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst_norm = np.uint8(dst_norm)
    _, _, _, centroids = cv2.connectedComponentsWithStats(dst_norm)

    # 亚像素角点
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(np.float32(gray), np.float32(centroids[1:]), (5, 5), (-1, -1), criteria)

    # 简单分类（基于局部梯度）
    result = original.copy()

    for corner in corners:
        x, y = int(corner[0]), int(corner[1])

        # 提取局部区域
        patch = gray[max(0, y - 10):min(gray.shape[0], y + 10),
                max(0, x - 10):min(gray.shape[1], x + 10)]

        if patch.size < 100:
            continue

        # 计算梯度方向
        sobelx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(sobelx, sobely)

        # 统计主要方向
        ang_bins = np.zeros(36)
        for i in range(36):
            mask = (ang >= i * 10 * np.pi / 180) & (ang < (i + 1) * 10 * np.pi / 180) & (mag > np.mean(mag) * 0.5)
            ang_bins[i] = np.sum(mask)

        # 找峰值
        peaks = np.sum(ang_bins > np.mean(ang_bins) * 1.5)

        if peaks >= 4:
            color, label = (0, 0, 255), 'X'  # 交叉点
        elif peaks == 3:
            color, label = (0, 255, 0), 'T'  # T型点
        else:
            color, label = (255, 0, 0), 'L'  # L型角点

        cv2.circle(result, (x, y), 5, color, -1)
        cv2.putText(result, label, (x - 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow('Junction Detection', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result


if __name__ == "__main__":
    # 使用示例
    detect_junctions(local_path("img.png"))
    # detect_junctions('Harris-LXT.png')
    # detect_junctions_simple('img.png')