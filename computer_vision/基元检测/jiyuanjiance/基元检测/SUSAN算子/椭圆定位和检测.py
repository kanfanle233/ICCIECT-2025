# -*- coding: utf-8 -*-
"""
教学示例：椭圆定位和检测

- 功能：演示 基元检测 中与“椭圆定位和检测”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""

import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def local_path(name: str) -> str:
    """返回与脚本同目录的资源路径，避免只能在某个固定工作目录下运行。"""
    return str(BASE_DIR / name)


def create_ellipse_test_image():
    """创建包含多种椭圆的测试图像"""
    img = np.zeros((500, 500, 3), dtype=np.uint8)

    # 水平椭圆
    cv2.ellipse(img, (150, 150), (80, 30), 0, 0, 360, (255, 255, 255), 2)

    # 垂直椭圆
    cv2.ellipse(img, (350, 150), (30, 80), 0, 0, 360, (255, 255, 255), 2)

    # 旋转45度椭圆
    cv2.ellipse(img, (150, 350), (70, 30), 45, 0, 360, (255, 255, 255), 2)

    # 旋转-30度椭圆
    cv2.ellipse(img, (350, 350), (60, 20), -30, 0, 360, (255, 255, 255), 2)

    # 添加一个圆（用于对比）
    cv2.circle(img, (250, 250), 50, (255, 255, 255), 2)

    return img


def detect_ellipses_by_fitting(image):
    """
    方法1: 基于轮廓拟合的椭圆检测

    参数:
        image: 输入图像

    返回:
        result: 标记了椭圆的图像
        ellipses: 检测到的椭圆参数列表 [(center, axes, angle), ...]
    """
    print("\n" + "=" * 60)
    print("方法1: 基于轮廓拟合的椭圆检测")
    print("=" * 60)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = image.copy()
    ellipses = []

    print(f"找到 {len(contours)} 个轮廓")

    for i, contour in enumerate(contours):
        # 过滤太小的轮廓
        if len(contour) < 5:
            continue

        # 拟合椭圆
        try:
            ellipse = cv2.fitEllipse(contour)
            (x, y), (a, b), angle = ellipse

            # 计算长短轴
            major = max(a, b)
            minor = min(a, b)

            # 过滤：椭圆的长短轴不能太接近（排除圆）
            if major / minor > 1.2:
                ellipses.append(ellipse)

                # 画椭圆（蓝色）
                cv2.ellipse(result, ellipse, (255, 0, 0), 2)

                # 画中心点（红色）
                center = (int(x), int(y))
                cv2.circle(result, center, 3, (0, 0, 255), -1)

                # 标注长短轴和角度
                cv2.putText(result, f"{int(major)}x{int(minor)}",
                            (int(x) - 40, int(y) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        except:
            continue

    print(f"检测到 {len(ellipses)} 个椭圆")
    return result, ellipses


def detect_ellipses_by_hough_like(image):
    """
    方法2: 基于边缘点和方向的椭圆检测（模拟哈夫变换思想）
    """
    print("\n" + "=" * 60)
    print("方法2: 基于边缘点的椭圆检测")
    print("=" * 60)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = image.copy()
    ellipses = []

    print(f"边缘点数量: {np.sum(edges > 0)}")
    print(f"找到 {len(contours)} 个轮廓")

    for contour in contours:
        if len(contour) >= 5:
            try:
                # 拟合椭圆
                ellipse = cv2.fitEllipse(contour)
                (x, y), (a, b), angle = ellipse

                # 计算拟合误差
                error = calculate_fitting_error(contour, ellipse)

                # 过滤：拟合误差小的才是好椭圆
                if error < 1.0:
                    ellipses.append(ellipse)

                    # 画椭圆（蓝色）
                    cv2.ellipse(result, ellipse, (255, 0, 0), 2)

                    # 画中心点（红色）
                    center = (int(x), int(y))
                    cv2.circle(result, center, 3, (0, 0, 255), -1)

                    # 显示拟合误差
                    cv2.putText(result, f"err:{error:.2f}",
                                (int(x) - 30, int(y) - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            except:
                continue

    print(f"检测到 {len(ellipses)} 个椭圆")
    return result, ellipses


def calculate_fitting_error(contour, ellipse):
    """
    计算轮廓与拟合椭圆的误差
    """
    (x, y), (a, b), angle = ellipse
    angle_rad = np.deg2rad(angle)

    total_error = 0
    for point in contour:
        px, py = point[0]

        # 将点转换到椭圆坐标系
        dx = px - x
        dy = py - y

        # 旋转
        rx = dx * np.cos(angle_rad) + dy * np.sin(angle_rad)
        ry = -dx * np.sin(angle_rad) + dy * np.cos(angle_rad)

        # 计算到椭圆边界的距离
        if a > 0 and b > 0:
            dist = np.sqrt((rx / (a / 2)) ** 2 + (ry / (b / 2)) ** 2) - 1
            total_error += abs(dist)

    return total_error / len(contour)


def detect_ellipses_by_color(image):
    """
    方法3: 基于颜色的椭圆检测（适用于彩色图像）
    """
    print("\n" + "=" * 60)
    print("方法3: 基于颜色的椭圆检测")
    print("=" * 60)

    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义颜色范围（这里以白色为例）
    lower = np.array([0, 0, 200])
    upper = np.array([180, 30, 255])

    # 创建掩码
    mask = cv2.inRange(hsv, lower, upper)

    # 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = image.copy()
    ellipses = []

    print(f"找到 {len(contours)} 个轮廓")

    for contour in contours:
        if len(contour) >= 5:
            area = cv2.contourArea(contour)
            if area < 100:  # 过滤小区域
                continue

            try:
                ellipse = cv2.fitEllipse(contour)
                ellipses.append(ellipse)

                # 画椭圆（绿色）
                cv2.ellipse(result, ellipse, (0, 255, 0), 2)

                # 画中心点（红色）
                (x, y), (a, b), angle = ellipse
                center = (int(x), int(y))
                cv2.circle(result, center, 3, (0, 0, 255), -1)

            except:
                continue

    print(f"检测到 {len(ellipses)} 个椭圆")
    return result, ellipses


def get_ellipse_parameters(ellipse):
    """
    提取椭圆参数
    """
    (x, y), (a, b), angle = ellipse

    return {
        'center': (int(x), int(y)),
        'major_axis': max(a, b),
        'minor_axis': min(a, b),
        'angle': angle,
        'area': np.pi * (a / 2) * (b / 2),
        ' eccentricity': np.sqrt(1 - (min(a, b) / max(a, b)) ** 2)
    }


def draw_ellipse_info(image, ellipse, color=(255, 255, 0)):
    """
    在图像上绘制椭圆详细信息
    """
    (x, y), (a, b), angle = ellipse
    center = (int(x), int(y))

    # 画椭圆
    cv2.ellipse(image, ellipse, color, 2)

    # 画中心
    cv2.circle(image, center, 3, (0, 0, 255), -1)

    # 画长短轴
    angle_rad = np.deg2rad(angle)
    major_end = (int(x + (a / 2) * np.cos(angle_rad)),
                 int(y + (a / 2) * np.sin(angle_rad)))
    minor_end = (int(x + (b / 2) * np.cos(angle_rad + np.pi / 2)),
                 int(y + (b / 2) * np.sin(angle_rad + np.pi / 2)))

    cv2.line(image, center, major_end, (0, 255, 0), 1)
    cv2.line(image, center, minor_end, (0, 255, 255), 1)

    # 标注参数
    info = f"a={a:.1f}, b={b:.1f}, angle={angle:.1f}"
    cv2.putText(image, info, (int(x) - 50, int(y) - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def main():
    """主函数"""

    # # 创建测试图像
    # print("创建测试图像...")
    # img = create_ellipse_test_image()
    # original = img.copy()

    # 或者读取真实图像
    img = cv2.imread(local_path("circle.png"))
    if img is None:
        print("使用测试图像")
        img = create_ellipse_test_image()
    original = img.copy()

    # 方法1: 轮廓拟合
    result1, ellipses1 = detect_ellipses_by_fitting(img)

    # 方法2: 边缘点检测
    result2, ellipses2 = detect_ellipses_by_hough_like(img)

    # 方法3: 颜色检测
    result3, ellipses3 = detect_ellipses_by_color(img)

    # 显示详细信息的图像
    detailed = original.copy()
    for ellipse in ellipses1:
        draw_ellipse_info(detailed, ellipse)

    # 显示结果
    # cv2.imshow('1. Original', original)
    # cv2.imshow('2. Contour Fitting', result1)
    cv2.imshow('3. Edge-based', result2)
    # cv2.imshow('4. Color-based', result3)
    # cv2.imshow('5. Detailed Info', detailed)

    # 打印椭圆参数
    print("\n" + "=" * 60)
    print("椭圆参数详情")
    print("=" * 60)

    for i, ellipse in enumerate(ellipses1):
        params = get_ellipse_parameters(ellipse)
        print(f"\n椭圆 {i + 1}:")
        print(f"  中心: {params['center']}")
        print(f"  长轴: {params['major_axis']:.1f}")
        print(f"  短轴: {params['minor_axis']:.1f}")
        print(f"  角度: {params['angle']:.1f}°")
        print(f"  面积: {params['area']:.1f}")
        print(f"  离心率: {params['eccentricity']:.3f}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 简化版本：直接检测并显示椭圆
def simple_ellipse_detection(image_path):
    """
    简化的椭圆检测函数
    """
    img = cv2.imread(image_path)
    if img is None:
        print("使用测试图像")
        img = create_ellipse_test_image()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img.copy()

    for contour in contours:
        if len(contour) >= 5:
            area = cv2.contourArea(contour)
            if area < 100:
                continue

            try:
                ellipse = cv2.fitEllipse(contour)
                (x, y), (a, b), angle = ellipse

                # 过滤掉圆（长短轴比例接近1）
                if max(a, b) / min(a, b) > 1.2:
                    cv2.ellipse(result, ellipse, (0, 255, 0), 2)
                    center = (int(x), int(y))
                    cv2.circle(result, center, 3, (0, 0, 255), -1)

            except:
                continue

    cv2.imshow('Ellipse Detection', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result


if __name__ == "__main__":
    # 运行完整版本
    main()

    # 或运行简化版本
    # simple_ellipse_detection('ellipse_image.png')
