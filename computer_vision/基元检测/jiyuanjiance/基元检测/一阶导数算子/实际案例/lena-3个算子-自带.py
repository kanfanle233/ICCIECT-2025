# -*- coding: utf-8 -*-
"""
教学示例：lena-3个算子-自带

- 功能：演示 一阶导数边缘检测 中与“lena-3个算子-自带”相关的核心流程。
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
# 读取图像
img = cv2.imread(local_path("lena.png"))
if img is None:
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (250, 250), (255, 255, 255), 2)
    cv2.circle(img, (150, 150), 80, (255, 255, 255), 2)

# 灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. Robert算子 (使用自定义核，因为OpenCV没有内置Robert)
kernel_robert_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
kernel_robert_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
robert_x = cv2.filter2D(gray, cv2.CV_32F, kernel_robert_x)
robert_y = cv2.filter2D(gray, cv2.CV_32F, kernel_robert_y)
robert = cv2.magnitude(robert_x, robert_y)
robert = cv2.normalize(robert, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 2. Prewitt算子 (使用自定义核)
kernel_prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
kernel_prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
prewitt_x = cv2.filter2D(gray, cv2.CV_32F, kernel_prewitt_x)
prewitt_y = cv2.filter2D(gray, cv2.CV_32F, kernel_prewitt_y)
prewitt = cv2.magnitude(prewitt_x, prewitt_y)
prewitt = cv2.normalize(prewitt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 3. Sobel算子 (OpenCV内置)
sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)
sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 显示四张图
cv2.imshow('Original', img)
cv2.imshow('Robert', robert)
cv2.imshow('Prewitt', prewitt)
cv2.imshow('Sobel', sobel)

cv2.waitKey(0)
cv2.destroyAllWindows()