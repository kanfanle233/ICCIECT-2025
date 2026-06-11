# -*- coding: utf-8 -*-
"""
教学示例：Hough-tansform-检测直线圆椭圆-manualdraw

- 功能：演示 基元检测 中与“Hough-tansform-检测直线圆椭圆-manualdraw”相关的核心流程。
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
def adaptive_hough_detection(image_path):
    """
    自适应哈夫变换检测（处理手绘图）
    """

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图像")
        return

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("=" * 50)
    print("自适应哈夫变换检测")
    print("=" * 50)

    # 步骤1: 图像预处理
    print("\n1. 图像预处理...")

    # 1.1 二值化
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 1.2 形态学操作 - 连接断点
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 1.3 去除小噪点
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # 1.4 边缘检测（使用更低的阈值）
    edges = cv2.Canny(cleaned, 30, 100)  # 降低阈值

    # 显示预处理结果
    cv2.imshow('Original', original)
    cv2.imshow('Binary', binary)
    cv2.imshow('Closed (断点连接)', closed)
    cv2.imshow('Cleaned (去噪)', cleaned)
    cv2.imshow('Edges', edges)

    # 步骤2: 直线检测（使用更宽松的参数）
    print("\n2. 直线检测...")

    # 尝试不同参数
    line_params = [
        (1, np.pi / 180, 80),  # 宽松参数
        (1, np.pi / 180, 60),  # 更宽松
        (1, np.pi / 180, 100),  # 标准
    ]

    best_lines = None
    best_count = 0

    for dp, theta, threshold in line_params:
        lines = cv2.HoughLines(edges, dp, theta, threshold)
        if lines is not None:
            count = len(lines)
            print(f"  参数(threshold={threshold}): 检测到 {count} 条直线")
            if count > best_count:
                best_count = count
                best_lines = lines
        else:
            print(f"  参数(threshold={threshold}): 未检测到直线")

    # 绘制检测到的直线
    line_result = original.copy()
    if best_lines is not None:
        print(f"\n最佳结果: 检测到 {best_count} 条直线")
        for line in best_lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(line_result, (x1, y1), (x2, y2), (0, 0, 255), 2)
    else:
        print("所有参数均未检测到直线")

    # 步骤3: 圆检测
    print("\n3. 圆检测...")

    blurred = cv2.medianBlur(cleaned, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=20,  # 降低阈值
        minRadius=5,
        maxRadius=200
    )

    circle_result = original.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(f"检测到 {circles.shape[1]} 个圆")
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            cv2.circle(circle_result, center, 1, (0, 0, 255), 3)
            cv2.circle(circle_result, center, radius, (0, 255, 0), 2)
    else:
        print("未检测到圆")

    # 显示结果
    cv2.imshow('Lines Detection', line_result)
    cv2.imshow('Circles Detection', circle_result)

    # 创建合并结果
    combined = original.copy()
    if best_lines is not None:
        for line in best_lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(combined, (x1, y1), (x2, y2), (0, 0, 255), 1)

    if circles is not None:
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            cv2.circle(combined, center, radius, (0, 255, 0), 1)

    cv2.imshow('Combined Result', combined)

    print("\n" + "=" * 50)
    print("检测完成，按任意键退出")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return best_lines, circles


def draw_perfect_shapes():
    """绘制完美的测试图形（用于对比）"""

    img = np.zeros((500, 500, 3), dtype=np.
                   uint8)

    # 使用cv2.line画直线（这是完美的直线）
    cv2.line(img, (50, 50), (450, 50), (255, 255, 255), 2)
    cv2.line(img, (50, 250), (450, 250), (255, 255, 255), 2)
    cv2.line(img, (50, 450), (450, 450), (255, 255, 255), 2)

    # 使用cv2.circle画圆（这是完美的圆）
    cv2.circle(img, (150, 150), 50, (255, 255, 255), 2)
    cv2.circle(img, (350, 150), 50, (255, 255, 255), 2)

    # 使用cv2.ellipse画椭圆
    cv2.ellipse(img, (250, 350), (80, 30), 0, 0, 360, (255, 255, 255), 2)

    # cv2.imwrite('perfect_shapes.png', img)
    print("完美测试图像已生成: perfect_shapes.png")
    return img


if __name__ == "__main__":
    # 先生成完美的测试图像
    perfect = draw_perfect_shapes()

    # 检测手绘图
    adaptive_hough_detection(local_path("manualdraw.jpg"))
