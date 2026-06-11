"""
教学示例：01一阶导数算子 lena 三个算子对比

- 功能：演示 一阶导数边缘检测 中与“01一阶导数算子 lena 三个算子对比”相关的核心流程。
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
# 1. 读取图像并直接转换为灰度图
# cv2.IMREAD_GRAYSCALE 参数确保读取出来的 img 是一个二维的灰度矩阵
img = cv2.imread(local_path("lena.png"), cv2.IMREAD_GRAYSCALE)

if img is None:
    print("错误：无法读取 'lena.png'。请检查图片是否在当前目录下。")
    exit()

# ================= 新增：在控制台打印灰度图矩阵 =================
print("=== 灰度图矩阵数据 (Pixel Values) ===")
print(img)
print("=========================================================")
print(f"当前图片（矩阵）的形状 (高度, 宽度): {img.shape}")
print(f"当前图片（矩阵）的数据类型: {img.dtype}")
print("说明: 矩阵中的每一个数字代表一个像素的灰度值 (0-255)。")
print("=========================================================\n")
# ===============================================================

# 2. Roberts 算子
kernel_Roberts_x = np.array([[1, 0], [0, -1]], dtype=int)
kernel_Roberts_y = np.array([[0, 1], [-1, 0]], dtype=int)

x_roberts = cv2.filter2D(img, cv2.CV_16S, kernel_Roberts_x)
y_roberts = cv2.filter2D(img, cv2.CV_16S, kernel_Roberts_y)

absX_roberts = cv2.convertScaleAbs(x_roberts)
absY_roberts = cv2.convertScaleAbs(y_roberts)
roberts_img = cv2.addWeighted(absX_roberts, 0.5, absY_roberts, 0.5, 0)

# 3. Prewitt 算子
kernel_Prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
kernel_Prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)

x_prewitt = cv2.filter2D(img, cv2.CV_16S, kernel_Prewitt_x)
y_prewitt = cv2.filter2D(img, cv2.CV_16S, kernel_Prewitt_y)

absX_prewitt = cv2.convertScaleAbs(x_prewitt)
absY_prewitt = cv2.convertScaleAbs(y_prewitt)
prewitt_img = cv2.addWeighted(absX_prewitt, 0.5, absY_prewitt, 0.5, 0)

# 4. Sobel 算子
x_sobel = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y_sobel = cv2.Sobel(img, cv2.CV_16S, 0, 1)

absX_sobel = cv2.convertScaleAbs(x_sobel)
absY_sobel = cv2.convertScaleAbs(y_sobel)
sobel_img = cv2.addWeighted(absX_sobel, 0.5, absY_sobel, 0.5, 0)

# 5. 使用 cv2.imshow 拼接显示
def add_label(image, text):
    """在图像左上角添加文字标签，返回带标签的副本。"""
    img_copy = image.copy()
    cv2.putText(img_copy, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2, cv2.LINE_AA)
    return img_copy

# 这里的 img 已经是灰度图了，添加标签作为原图参考
img_labeled = add_label(img, "Original (Grayscale)")
roberts_labeled = add_label(roberts_img, "Roberts")
prewitt_labeled = add_label(prewitt_img, "Prewitt")
sobel_labeled = add_label(sobel_img, "Sobel")

# 拼接图像
top_row = np.hstack((img_labeled, roberts_labeled))
bottom_row = np.hstack((prewitt_labeled, sobel_labeled))
combined_img = np.vstack((top_row, bottom_row))

# 弹出窗口显示
cv2.imshow('Edge Detection Comparison', combined_img)

print("请查看弹出的图片窗口，并按键盘上的任意键关闭窗口...")
cv2.waitKey(0)
cv2.destroyAllWindows()