"""
教学示例：edge detection

- 功能：演示 基元检测 中与“edge detection”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

def local_path(name: str) -> str:
    """返回脚本目录下的资源路径，避免依赖当前工作目录。"""
    return str(BASE_DIR / name)
# pip install opencv-python numpy matplotlib 提前安装依赖命令
import sys
import cv2
import numpy as np
import matplotlib

# 【关键步骤 1】设置后端为交互式窗口后端 (TkAgg 或 Qt5Agg)
try:
    import PyQt5

    matplotlib.use('Qt5Agg')
except ImportError:
    try:
        import tkinter

        matplotlib.use('TkAgg')
    except ImportError:
        pass

import matplotlib.pyplot as plt


def run_edge_detection(image_path):
    """读取灰度图像，分别用Roberts、Prewitt、Sobel、Laplacian和Canny进行边缘检测并对比显示。"""
    # 1. 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"错误：无法读取图像 {image_path}")
        return

    # --- 定义算子核 ---
    roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

    # --- 计算边缘 ---
    # Roberts
    rx = cv2.filter2D(image, cv2.CV_32F, roberts_x)
    ry = cv2.filter2D(image, cv2.CV_32F, roberts_y)
    res_roberts = np.clip(np.sqrt(rx ** 2 + ry ** 2), 0, 255).astype(np.uint8)

    # Prewitt
    px = cv2.filter2D(image, cv2.CV_32F, prewitt_x)
    py = cv2.filter2D(image, cv2.CV_32F, prewitt_y)
    res_prewitt = np.clip(np.sqrt(px ** 2 + py ** 2), 0, 255).astype(np.uint8)

    # Sobel
    sx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    res_sobel = np.clip(np.sqrt(sx ** 2 + sy ** 2), 0, 255).astype(np.uint8)

    # Laplacian
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    res_lap = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
    res_lap = np.clip(np.absolute(res_lap), 0, 255).astype(np.uint8)

    # Canny 边缘检测 ---
    # cv2.Canny(image, threshold1, threshold2)
    # threshold1: 低阈值，threshold2: 高阈值
    # 这里使用 50 和 150 作为常用示例值，可根据实际图像调整
    res_canny = cv2.Canny(image, 50, 150)

    # --- 绘图 ---
    plt.figure(figsize=(14, 10))

    # 标题和图像列表
    titles = ["Original", "Roberts", "Prewitt", "Sobel", "Laplacian", "Canny"]
    images = [image, res_roberts, res_prewitt, res_sobel, res_lap, res_canny]

    # 循环绘制 6 个子图
    # 布局为 2行3列，索引从 1 到 6
    # 第二行最后一列 对应的正是索引 6
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i], fontsize=12)
        plt.axis('off')

    plt.tight_layout()

    # 【关键步骤 3】显示窗口
    plt.show(block=True)


if __name__ == "__main__":
    target_image = local_path("11.jpg")

    if cv2.imread(target_image) is None:
        print("未找到指定图片，正在生成一张测试用图片...")
        img = np.zeros((512, 512), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (200, 200), 255, -1)
        cv2.circle(img, (350, 350), 100, 255, -1)
        cv2.line(img, (50, 400), (450, 400), 255, 2)
        cv2.imwrite(target_image, img)
        print(f" 测试图片已生成：{target_image}")

    run_edge_detection(target_image)