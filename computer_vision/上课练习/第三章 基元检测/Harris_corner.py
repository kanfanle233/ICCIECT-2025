"""
教学示例：Harris corner

- 功能：演示 基元检测 中与“Harris corner”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


BASE_DIR = Path(__file__).resolve().parent

def local_path(name: str) -> str:
    """返回脚本目录下的资源路径，避免依赖当前工作目录。"""
    return str(BASE_DIR / name)
# 【关键】强制弹出独立窗口
try:
    import tkinter

    plt.switch_backend('TkAgg')
except:
    pass

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def detect_harris_corners(image_path):
    """读取图像并执行Harris角点检测，展示原图、响应热力图、角点标记和Canny对比。"""
    if not os.path.exists(image_path):
        print(f"错误：找不到 {image_path}")
        return

    # 1. 读取图像
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("无法读取图片")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Harris 角点检测参数
    # blockSize: 邻域大小 (考虑多大的范围)
    # ksize: Sobel 算子孔径 (用于计算梯度)
    # k: 灵敏度系数 (0.04 ~ 0.06 是经验值，越小越敏感)
    block_size = 2
    ksize = 3
    k = 0.04

    # 执行 Harris 检测
    # dst 是一个浮点数矩阵，每个像素的值代表它是角点的“可能性”(响应值 R)
    dst = cv2.cornerHarris(gray, block_size, ksize, k)

    # 3. 结果可视化处理

    # A. 归一化响应图 (0-255)，方便肉眼观察哪里响应强
    dst_norm = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX)
    dst_norm_uint8 = np.uint8(dst_norm)

    # B. 提取具体的角点坐标 (阈值过滤 + 非极大值抑制的简化版)
    # 设定阈值：只保留响应值大于 最大响应值 1% 的点
    threshold = 0.01 * dst.max()

    # 创建副本用于画点
    img_with_corners = img_rgb.copy()

    corner_points = []
    # 遍历所有像素，找出超过阈值的点
    # 注意：这里没有做严格的非极大值抑制，所以相邻像素可能都被选中
    # 为了视觉效果，我们通常结合 dilation 或直接画圈
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            if dst[i, j] > threshold:
                corner_points.append((j, i))  # (x, y)
                # 在图上画一个小红点
                cv2.circle(img_with_corners, (j, i), 3, (255, 0, 0), -1)

    print(f"检测到 {len(corner_points)} 个潜在角点。")

    # 4. 绘图展示 (2行2列)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.canvas.manager.set_window_title('Harris 角点检测演示')
    fig.suptitle('Harris Interest Point Operator (哈里斯兴趣点算子)', fontsize=16, fontweight='bold')

    # 图1: 原图
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('1. 原始图像')
    axes[0, 0].axis('off')

    # 图2: Harris 响应热力图 (越白表示越像角点)
    axes[0, 1].imshow(dst_norm_uint8, cmap='hot')
    axes[0, 1].set_title(f'2. Harris 响应热力图 (R值)\n(白色区域为高响应区)')
    axes[0, 1].axis('off')

    # 图3: 标记了角点的结果图
    axes[1, 0].imshow(img_with_corners)
    axes[1, 0].set_title(f'3. 检测到的角点 (红点)\n(阈值: {threshold:.2f})')
    axes[1, 0].axis('off')

    # 图4: 局部放大示意 (可选，这里放一个边缘检测对比作为参考)
    edges = cv2.Canny(gray, 50, 150)
    axes[1, 1].imshow(edges, cmap='gray')
    axes[1, 1].set_title('4. Canny 边缘检测 (对比)\n(边缘 vs 角点)')
    axes[1, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    # 确保你有 test.jpg，或者修改这里的文件名
    detect_harris_corners(local_path("11.jpg"))