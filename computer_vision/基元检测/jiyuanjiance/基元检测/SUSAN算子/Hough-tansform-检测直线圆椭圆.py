# -*- coding: utf-8 -*-
"""
教学示例：Hough-tansform-检测直线圆椭圆

- 功能：演示 基元检测 中与“Hough-tansform-检测直线圆椭圆”相关的核心流程。
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
def create_test_image():
    """创建包含直线、圆、椭圆的测试图像"""
    img = np.zeros((500, 500, 3), dtype=np.uint8)

    # 添加直线
    cv2.line(img, (50, 50), (450, 50), (255, 255, 255), 2)  # 水平线
    cv2.line(img, (50, 250), (450, 250), (255, 255, 255), 2)  # 水平线
    cv2.line(img, (50, 450), (450, 450), (255, 255, 255), 2)  # 水平线
    cv2.line(img, (50, 50), (50, 450), (255, 255, 255), 2)  # 垂直线
    cv2.line(img, (250, 50), (250, 450), (255, 255, 255), 2)  # 垂直线
    cv2.line(img, (450, 50), (450, 450), (255, 255, 255), 2)  # 垂直线
    cv2.line(img, (50, 50), (450, 450), (255, 255, 255), 2)  # 对角线
    cv2.line(img, (450, 50), (50, 450), (255, 255, 255), 2)  # 对角线

    # 添加圆
    cv2.circle(img, (150, 150), 60, (255, 255, 255), 2)
    cv2.circle(img, (350, 150), 60, (255, 255, 255), 2)
    cv2.circle(img, (150, 350), 60, (255, 255, 255), 2)
    cv2.circle(img, (350, 350), 60, (255, 255, 255), 2)

    # 添加椭圆
    cv2.ellipse(img, (150, 150), (80, 40), 0, 0, 360, (255, 255, 255), 2)  # 水平椭圆
    cv2.ellipse(img, (350, 150), (40, 80), 0, 0, 360, (255, 255, 255), 2)  # 垂直椭圆
    cv2.ellipse(img, (150, 350), (70, 30), 45, 0, 360, (255, 255, 255), 2)  # 旋转45度椭圆
    cv2.ellipse(img, (350, 350), (60, 20), -30, 0, 360, (255, 255, 255), 2)  # 旋转-30度椭圆

    return img


def detect_lines(image):
    """
    函数1: 哈夫变换检测直线

    参数:
        image: 输入图像

    返回:
        result: 标记了直线的图像
        lines: 检测到的直线参数
    """
    print("\n" + "=" * 60)
    print("函数1: 哈夫直线检测")
    print("=" * 60)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 标准哈夫变换检测直线
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

    result = image.copy()

    if lines is not None:
        print(f"检测到 {len(lines)} 条直线")
        for i, line in enumerate(lines):
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # 用不同颜色画直线
            color = (0, 0, 255)  # 红色
            cv2.line(result, (x1, y1), (x2, y2), color, 2)
    else:
        print("未检测到直线")

    return result, lines


def detect_circles(image):
    """
    函数2: 哈夫变换检测圆

    参数:
        image: 输入图像

    返回:
        result: 标记了圆的图像
        circles: 检测到的圆参数
    """
    print("\n" + "=" * 60)
    print("函数2: 哈夫圆检测")
    print("=" * 60)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    # 哈夫圆检测
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=10, maxRadius=100)
    # #
    # circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 30,
    #                            param1=100, param2=50,  # 提高阈值
    #                            minRadius=15, maxRadius=150)

    result = image.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))
        num_circles = circles.shape[1]
        print(f"检测到 {num_circles} 个圆")

        for i, circle in enumerate(circles[0, :]):
            center = (circle[0], circle[1])
            radius = circle[2]

            # 画圆心（红色）
            cv2.circle(result, center, 1, (0, 0, 255), 3)
            # 画圆（绿色）
            cv2.circle(result, center, radius, (0, 255, 0), 2)
    else:
        print("未检测到圆")
        circles = None

    return result, circles


def detect_ellipses(image):
    """
    函数3: 检测椭圆（OpenCV没有直接的哈夫椭圆变换，使用轮廓拟合）

    参数:
        image: 输入图像

    返回:
        result: 标记了椭圆的图像
        ellipses: 检测到的椭圆参数
    """
    print("\n" + "=" * 60)
    print("函数3: 椭圆检测（基于轮廓拟合）")
    print("=" * 60)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
            ellipses.append(ellipse)

            # 画椭圆（蓝色）
            cv2.ellipse(result, ellipse, (255, 0, 0), 2)

            # 画中心点（红色）
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            cv2.circle(result, center, 3, (0, 0, 255), -1)

        except:
            continue

    print(f"检测到 {len(ellipses)} 个椭圆")

    return result, ellipses


def detect_ellipses_hough_like(image):
    """
    函数3(备选): 模拟哈夫变换思想的椭圆检测
    （使用边缘点的方向和位置投票）
    """
    print("\n" + "=" * 60)
    print("函数3(备选): 基于哈夫思想的椭圆检测")
    print("=" * 60)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 获取边缘点
    points = np.argwhere(edges > 0)
    print(f"边缘点数量: {len(points)}")

    # 简化的椭圆检测：使用OpenCV的轮廓拟合
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = image.copy()
    ellipses = []

    for contour in contours:
        if len(contour) >= 5:  # 椭圆拟合至少需要5个点
            try:
                ellipse = cv2.fitEllipse(contour)

                # 过滤太小的椭圆
                (x, y), (a, b), angle = ellipse
                if a > 20 and b > 20:
                    ellipses.append(ellipse)
                    cv2.ellipse(result, ellipse, (0, 255, 255), 2)  # 黄色
                    center = (int(x), int(y))
                    cv2.circle(result, center, 3, (0, 0, 255), -1)
            except:
                continue

    print(f"检测到 {len(ellipses)} 个椭圆")
    return result, ellipses


def main():
    """主函数 - 调用三个检测函数"""

    # # 创建测试图像
    # print("创建测试图像...")
    # img = create_test_image()
    # original = img.copy()
    # 读取图像
    img = cv2.imread(local_path("Hough-transform.png"))
    # img = cv2.imread('circle.png')
    if img is None:
        print(f"错误：无法读取图像 {local_path('Hough-transform.png')}")
        return

    original = img.copy()
    print(f"图像尺寸: {img.shape}")



    # 显示原始图像
    cv2.imshow('0. 原始图像', original)

    # 1. 调用直线检测函数
    line_result, lines = detect_lines(img)
    cv2.imshow('1. 直线检测', line_result)

    # 2. 调用圆检测函数
    circle_result, circles = detect_circles(img)
    cv2.imshow('2. 圆检测', circle_result)

    # 3. 调用椭圆检测函数
    ellipse_result, ellipses = detect_ellipses(img)
    cv2.imshow('3. 椭圆检测', ellipse_result)

    # 可选：使用备选椭圆检测
    # ellipse_result2, ellipses2 = detect_ellipses_hough_like(img)
    # cv2.imshow('3b. 椭圆检测(备选)', ellipse_result2)

    # 创建合并结果图
    combined = original.copy()

    # 画直线（红色）
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(combined, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # 画圆（绿色）
    if circles is not None:
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            cv2.circle(combined, center, radius, (0, 255, 0), 1)

    # 画椭圆（蓝色）
    for ellipse in ellipses:
        cv2.ellipse(combined, ellipse, (255, 0, 0), 1)

    cv2.imshow('4. 合并结果', combined)

    print("\n" + "=" * 60)
    print("检测完成，按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ## 保存结果
    # cv2.imwrite('original.jpg', original)
    # cv2.imwrite('lines_detection.jpg', line_result)
    # cv2.imwrite('circles_detection.jpg', circle_result)
    # cv2.imwrite('ellipses_detection.jpg', ellipse_result)
    # cv2.imwrite('combined_detection.jpg', combined)
    # print("\n结果已保存")


def print_detection_info(lines, circles, ellipses):
    """打印检测信息"""
    print("\n" + "=" * 60)
    print("检测结果汇总")
    print("=" * 60)

    print(f"直线: {0 if lines is None else len(lines)} 条")
    print(f"圆: {0 if circles is None else circles.shape[1]} 个")
    print(f"椭圆: {len(ellipses)} 个")


if __name__ == "__main__":
    main()
