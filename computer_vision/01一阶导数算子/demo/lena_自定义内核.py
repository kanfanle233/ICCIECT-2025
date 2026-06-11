"""
教学示例：lena 自定义内核

- 功能：演示 一阶导数边缘检测 中与“lena 自定义内核”相关的核心流程。
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
# 1. 读取并打印图像矩阵
img = cv2.imread(local_path("lena.png"), cv2.IMREAD_GRAYSCALE)

if img is None:
    print("错误：无法读取 'lena.png'。请检查图片是否在当前目录下。")
    exit()

print("=== 原图部分灰度矩阵 ===")
print(img)
print("======================\n")

# -------------------------------------------------------------
# 核心：完全使用 numpy 自定义内核
# -------------------------------------------------------------

# 2. Roberts 算子内核 (2x2)
# X方向：寻找对角线边缘
kernel_Roberts_x = np.array([[1, 0],
                             [0, -1]], dtype=np.float32)
# Y方向：寻找另一条对角线边缘
kernel_Roberts_y = np.array([[0, 1],
                             [-1, 0]], dtype=np.float32)

# 3. Prewitt 算子内核 (3x3)
# X方向：检测垂直边缘
kernel_Prewitt_x = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]], dtype=np.float32)
# Y方向：检测水平边缘
kernel_Prewitt_y = np.array([[-1, -1, -1],
                             [0, 0, 0],
                             [1, 1, 1]], dtype=np.float32)

# 4. Sobel 算子内核 (3x3)
# 注意看与 Prewitt 的区别：距离中心越近，权重越大 (2 和 -2)
kernel_Sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)
# Y方向
kernel_Sobel_y = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=np.float32)


# -------------------------------------------------------------
# 统一使用 cv2.filter2D 执行二维卷积
# -------------------------------------------------------------

def apply_custom_kernel(image, kernel_x, kernel_y):
    """使用自定义的X方向和Y方向卷积核对图像进行滤波，并合并梯度结果。"""
    # 使用 cv2.CV_16S (16位有符号整数) 防止计算过程中出现负数时被截断为0
    grad_x = cv2.filter2D(image, cv2.CV_16S, kernel_x)
    grad_y = cv2.filter2D(image, cv2.CV_16S, kernel_y)

    # 取绝对值并转回 uint8 显示格式
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # 将X方向和Y方向的梯度按照 1:1 的权重合并
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


# 应用自定义内核
roberts_img = apply_custom_kernel(img, kernel_Roberts_x, kernel_Roberts_y)
prewitt_img = apply_custom_kernel(img, kernel_Prewitt_x, kernel_Prewitt_y)
sobel_img = apply_custom_kernel(img, kernel_Sobel_x, kernel_Sobel_y)


# -------------------------------------------------------------
# 拼接与显示 (与之前相同)
# -------------------------------------------------------------

def add_label(image, text):
    """在图像左上角添加文字标签，返回带标签的副本。"""
    img_copy = image.copy()
    cv2.putText(img_copy, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2, cv2.LINE_AA)
    return img_copy


img_labeled = add_label(img, "Original (Grayscale)")
roberts_labeled = add_label(roberts_img, "Roberts (Custom)")
prewitt_labeled = add_label(prewitt_img, "Prewitt (Custom)")
sobel_labeled = add_label(sobel_img, "Sobel (Custom)")

top_row = np.hstack((img_labeled, roberts_labeled))
bottom_row = np.hstack((prewitt_labeled, sobel_labeled))
combined_img = np.vstack((top_row, bottom_row))

cv2.imshow('Custom Kernels Comparison', combined_img)

print("请查看弹出的图片窗口，并按键盘上的任意键关闭窗口...")
cv2.waitKey(0)
cv2.destroyAllWindows()