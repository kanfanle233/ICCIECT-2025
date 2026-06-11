"""
教学示例：02二阶导数算子 拉普拉斯 马尔 canny算子

- 功能：演示 二阶导数与边缘检测 中与“02二阶导数算子 拉普拉斯 马尔 canny算子”相关的核心流程。
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
# 1. 读取原图 (灰度)
img = cv2.imread(local_path("lena.png"), cv2.IMREAD_GRAYSCALE)

if img is None:
    print("错误：无法读取 'lena.png'。请检查图片是否在同一目录下。")
    exit()

# -------------------------------------------------------------
# 2. 调用 OpenCV 内置 API 进行边缘检测
# -------------------------------------------------------------

# 【算子1】Laplacian (拉普拉斯)
# 直接调用 cv2.Laplacian，使用 3x3 内核，数据类型 CV_16S 防截断
laplacian_16s = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
laplacian_img = cv2.convertScaleAbs(laplacian_16s)

# 【算子2】Marr-Hildreth (马尔算子 / LoG - Laplacian of Gaussian)
# 原理：先用高斯滤波平滑去噪，再用拉普拉斯求二阶导
blurred_img = cv2.GaussianBlur(img, (5, 5), 0)  # 5x5 高斯核去噪
marr_16s = cv2.Laplacian(blurred_img, cv2.CV_16S, ksize=3)
marr_img = cv2.convertScaleAbs(marr_16s)

# 【算子3】Canny 算子
# 原理：高斯去噪 -> 计算梯度 -> 非极大值抑制 -> 双阈值检测
# Canny 函数直接返回 uint8 格式的二值化边缘图，非常省事
canny_img = cv2.Canny(img, threshold1=50, threshold2=150)


# -------------------------------------------------------------
# 3. 绘制标签与特性面板
# -------------------------------------------------------------

def add_feature_labels(image, title, features):
    """在图像上绘制标题和特性说明列表，返回带标注的彩色图像。"""
    # 如果是单通道灰度图，转为 BGR 彩色图以便绘制彩色文字（尽管底图仍是黑白）
    # 这样可以避免某些情况下文字颜色显示异常
    img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()

    def draw_text_with_outline(img_target, text, pos, font_scale, color, thickness):
        """绘制带黑色描边的文字，提高在浅色背景上的可读性。"""
        cv2.putText(img_target, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img_target, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    draw_text_with_outline(img_bgr, title, (15, 35), 0.8, (255, 255, 255), 2)
    y_offset = 65
    for feature in features:
        draw_text_with_outline(img_bgr, f"- {feature}", (15, y_offset), 0.5, (220, 220, 220), 1)
        y_offset += 25
    return img_bgr


# 定义特性列表
original_features = ["Input Image", "8-bit Grayscale"]

laplacian_features = [
    "2nd Order Deriv",
    "Isotropic (All Dir)",
    "High Noise Sens",
    "Zero-Crossing",
    "cv2.Laplacian()"
]

marr_features = [
    "LoG (Laplacian+Gauss)",
    "Noise Reduced",
    "Smoother Edges",
    "Zero-Crossing",
    "Blur + Laplacian"
]

canny_features = [
    "Multi-Stage Optimal",
    "Non-Max Suppression",
    "Double Thresholds",
    "Thin/Crisp Edges",
    "cv2.Canny()"
]

# 给各图像贴上信息
img_labeled = add_feature_labels(img, "Original", original_features)
laplacian_labeled = add_feature_labels(laplacian_img, "Laplacian", laplacian_features)
marr_labeled = add_feature_labels(marr_img, "Marr-Hildreth (LoG)", marr_features)
canny_labeled = add_feature_labels(canny_img, "Canny", canny_features)

# -------------------------------------------------------------
# 4. 拼接并显示
# -------------------------------------------------------------

top_row = np.hstack((img_labeled, laplacian_labeled))
bottom_row = np.hstack((marr_labeled, canny_labeled))
combined_img = np.vstack((top_row, bottom_row))

cv2.imshow('Advanced Edge Detectors Comparison', combined_img)

print("高级算子面板已生成！请查看图片窗口，按任意键关闭...")
cv2.waitKey(0)
cv2.destroyAllWindows()