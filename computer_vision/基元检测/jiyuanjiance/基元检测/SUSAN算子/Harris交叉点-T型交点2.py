# -*- coding: utf-8 -*-
"""
教学示例：Harris交叉点-T型交点2

- 功能：演示 基元检测 中与“Harris交叉点-T型交点2”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""

import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def local_path(name: str) -> str:
    """返回脚本目录下的资源路径，保证直接运行脚本时也能找到示例图片。"""
    return str(BASE_DIR / name)


def detect_cross_points(image, harris_response, threshold=0.01):
    """
    检测交叉点 (X型) - 红色

    参数:
        image: 原始图像
        harris_response: Harris响应矩阵
        threshold: 角点阈值

    返回:
        cross_points: 交叉点坐标列表
        result_img: 标记了交叉点的图像
    """
    # 获取角点坐标
    corner_coords = np.argwhere(harris_response > threshold * harris_response.max())

    cross_points = []
    window_size = 15
    half = window_size // 2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    for y, x in corner_coords:
        # 提取局部区域
        y1, y2 = max(0, y - half), min(gray.shape[0], y + half + 1)
        x1, x2 = max(0, x - half), min(gray.shape[1], x + half + 1)
        patch = gray[y1:y2, x1:x2]

        if patch.size < 100:
            continue

        # 计算梯度方向
        sobelx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        direction = np.arctan2(sobely, sobelx) * 180 / np.pi

        # 梯度方向直方图
        mag_threshold = np.mean(magnitude) * 0.5
        strong_pixels = magnitude > mag_threshold

        if np.sum(strong_pixels) < 10:
            continue

        hist, _ = np.histogram(direction[strong_pixels], bins=36, range=(-180, 180))
        hist = np.convolve(hist, [1, 2, 1], mode='same')

        # 找峰值
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > np.mean(hist) * 1.2:
                peaks.append(i)

        # 交叉点：4个或以上峰值
        if len(peaks) >= 4:
            cross_points.append((x, y))

    # 标记交叉点
    result_img = image.copy()
    for x, y in cross_points:
        cv2.circle(result_img, (x, y), 7, (0, 0, 255), -1)  # 红色
        cv2.putText(result_img, 'X', (x - 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    print(f"检测到交叉点: {len(cross_points)}")
    return cross_points, result_img


def detect_t_junctions(image, harris_response, threshold=0.01):
    """
    检测T型交点 - 黑色

    参数:
        image: 原始图像
        harris_response: Harris响应矩阵
        threshold: 角点阈值

    返回:
        t_junctions: T型交点坐标列表
        result_img: 标记了T型交点的图像
    """
    # 获取角点坐标
    corner_coords = np.argwhere(harris_response > threshold * harris_response.max())

    t_junctions = []
    window_size = 15
    half = window_size // 2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    for y, x in corner_coords:
        # 提取局部区域
        y1, y2 = max(0, y - half), min(gray.shape[0], y + half + 1)
        x1, x2 = max(0, x - half), min(gray.shape[1], x + half + 1)
        patch = gray[y1:y2, x1:x2]

        if patch.size < 100:
            continue

        # 计算梯度方向
        sobelx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        direction = np.arctan2(sobely, sobelx) * 180 / np.pi

        # 梯度方向直方图
        mag_threshold = np.mean(magnitude) * 0.5
        strong_pixels = magnitude > mag_threshold

        if np.sum(strong_pixels) < 10:
            continue

        hist, _ = np.histogram(direction[strong_pixels], bins=36, range=(-180, 180))
        hist = np.convolve(hist, [1, 2, 1], mode='same')

        # 找峰值
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > np.mean(hist) * 1.2:
                peaks.append(i)

        # T型交点：3个峰值
        if len(peaks) == 3:
            t_junctions.append((x, y))

    # 标记T型交点
    result_img = image.copy()
    for x, y in t_junctions:
        cv2.circle(result_img, (x, y), 7, (0, 0, 0), -1)  # 黑色
        cv2.putText(result_img, 'T', (x - 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    print(f"检测到T型交点: {len(t_junctions)}")
    return t_junctions, result_img


def detect_corners(image, harris_response, threshold=0.01):
    """
    检测L型角点 - 蓝色

    参数:
        image: 原始图像
        harris_response: Harris响应矩阵
        threshold: 角点阈值

    返回:
        corners: L型角点坐标列表
        result_img: 标记了L型角点的图像
    """
    # 获取角点坐标
    corner_coords = np.argwhere(harris_response > threshold * harris_response.max())

    l_corners = []
    window_size = 15
    half = window_size // 2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    for y, x in corner_coords:
        # 提取局部区域
        y1, y2 = max(0, y - half), min(gray.shape[0], y + half + 1)
        x1, x2 = max(0, x - half), min(gray.shape[1], x + half + 1)
        patch = gray[y1:y2, x1:x2]

        if patch.size < 100:
            continue

        # 计算梯度方向
        sobelx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        direction = np.arctan2(sobely, sobelx) * 180 / np.pi

        # 梯度方向直方图
        mag_threshold = np.mean(magnitude) * 0.5
        strong_pixels = magnitude > mag_threshold

        if np.sum(strong_pixels) < 10:
            continue

        hist, _ = np.histogram(direction[strong_pixels], bins=36, range=(-180, 180))
        hist = np.convolve(hist, [1, 2, 1], mode='same')

        # 找峰值
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > np.mean(hist) * 1.2:
                peaks.append(i)

        # L型角点：2个峰值
        if len(peaks) == 2:
            l_corners.append((x, y))

    # 标记L型角点
    result_img = image.copy()
    for x, y in l_corners:
        cv2.circle(result_img, (x, y), 5, (255, 0, 0), -1)  # 蓝色
        cv2.putText(result_img, 'L', (x - 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    print(f"检测到L型角点: {len(l_corners)}")
    return l_corners, result_img


def create_test_image():
    """创建包含各种类型的测试图像"""
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    # X型交叉点 (红色)
    cv2.line(img, (100, 100), (200, 200), (255, 255, 255), 2)
    cv2.line(img, (200, 100), (100, 200), (255, 255, 255), 2)

    # T型交点 (黑色)
    cv2.line(img, (300, 100), (300, 200), (255, 255, 255), 2)
    cv2.line(img, (250, 150), (350, 150), (255, 255, 255), 2)

    # L型角点 (蓝色)
    cv2.rectangle(img, (50, 250), (150, 350), (255, 255, 255), 2)

    # 添加标签
    cv2.putText(img, 'X', (130, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, 'T', (330, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, 'L', (80, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def main(image_path=None):
    """主函数 - 调用三个检测函数"""

    # 读取或创建图像
    if image_path and cv2.imread(image_path) is not None:
        img = cv2.imread(image_path)
        print(f"读取图像: {image_path}")
    else:
        img = create_test_image()
        print("使用测试图像")

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Harris角点检测
    print("\n正在进行Harris角点检测...")
    gray_float = np.float32(gray)
    harris_response = cv2.cornerHarris(gray_float, 2, 3, 0.04)
    harris_response = cv2.dilate(harris_response, None)

    # 调用三个检测函数
    print("\n" + "=" * 40)
    cross_points, img_cross = detect_cross_points(img, harris_response)
    t_junctions, img_t = detect_t_junctions(img, harris_response)
    corners, img_corner = detect_corners(img, harris_response)
    print("=" * 40)

    # 合并结果
    result = original.copy()

    # 蓝色: L型角点
    for x, y in corners:
        cv2.circle(result, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(result, 'L', (x - 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 黑色: T型交点
    for x, y in t_junctions:
        cv2.circle(result, (x, y), 7, (0, 0, 0), -1)
        cv2.putText(result, 'T', (x - 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 红色: X型交叉点
    for x, y in cross_points:
        cv2.circle(result, (x, y), 7, (0, 0, 255), -1)
        cv2.putText(result, 'X', (x - 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 显示结果
    cv2.imshow('Original', original)
    cv2.imshow('Cross Points (X) - Red', img_cross)
    cv2.imshow('T Junctions - Black', img_t)
    cv2.imshow('L Corners - Blue', img_corner)
    cv2.imshow('Combined Result', result)

    # 显示Harris响应图
    harris_norm = cv2.normalize(harris_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    harris_color = cv2.applyColorMap(harris_norm, cv2.COLORMAP_HOT)
    cv2.imshow('Harris Response', harris_color)

    print("\n检测结果汇总:")
    print(f"L型角点 (蓝色): {len(corners)}")
    print(f"T型交点 (黑色): {len(t_junctions)}")
    print(f"X型交叉点 (红色): {len(cross_points)}")
    print(f"总计: {len(corners) + len(t_junctions) + len(cross_points)}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果
    # cv2.imwrite('cross_points.jpg', img_cross)
    # cv2.imwrite('t_junctions.jpg', img_t)
    # cv2.imwrite('corners.jpg', img_corner)
    # cv2.imwrite('combined_result.jpg', result)
    # print("\n结果已保存")


if __name__ == "__main__":
    # 使用示例: main('your_image.jpg') 或 main() 使用测试图像
    # main('img.png')
    main(local_path("Harris-LXT.png"))
