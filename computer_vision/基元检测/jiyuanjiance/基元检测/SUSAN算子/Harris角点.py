# -*- coding: utf-8 -*-
"""
教学示例：Harris角点

- 功能：演示 基元检测 中与“Harris角点”相关的核心流程。
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
img = cv2.imread(local_path("img.png"))
if img is None:
    # 创建测试图像
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (250, 250), (255, 255, 255), 2)
    cv2.line(img, (150, 50), (150, 250), (255, 255, 255), 2)

original = img.copy()

# Harris角点检测
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)

# 标记角点
img[dst > 0.01 * dst.max()] = [0, 0, 255]

# 显示
cv2.imshow('Original', original)
cv2.imshow('Harris Corners', img)
cv2.waitKey(0)
cv2.destroyAllWindows()