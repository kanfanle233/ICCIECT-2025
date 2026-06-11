# -*- coding: utf-8 -*-
"""
教学示例：边界细化

- 功能：演示 基元检测 中与“边界细化”相关的核心流程。
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
img = cv2.imread(local_path("lena.png"), cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (250, 250), 255, 3)
    cv2.circle(img, (150, 150), 80, 255, 3)

# 高斯平滑
blurred = cv2.GaussianBlur(img, (5, 5), 1.4)

# 计算梯度
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0)
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1)

# 梯度幅值和方向
mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
angle = np.arctan2(sobel_y, sobel_x)
angle = np.rad2deg(angle) % 180

# 非极大值抑制（边界细化）
h, w = mag.shape
refined = np.zeros((h, w), dtype=np.uint8)

for i in range(1, h - 1):
    for j in range(1, w - 1):
        current = mag[i, j]
        if current == 0:
            continue

        # 根据梯度方向比较邻居
        if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
            n1, n2 = mag[i, j - 1], mag[i, j + 1]  # 左右
        elif 22.5 <= angle[i, j] < 67.5:
            n1, n2 = mag[i - 1, j + 1], mag[i + 1, j - 1]  # 45°
        elif 67.5 <= angle[i, j] < 112.5:
            n1, n2 = mag[i - 1, j], mag[i + 1, j]  # 上下
        else:
            n1, n2 = mag[i - 1, j - 1], mag[i + 1, j + 1]  # 135°

        if current >= n1 and current >= n2:
            refined[i, j] = current

# 二值化
_, binary = cv2.threshold(refined, 30, 255, cv2.THRESH_BINARY)

# 显示
cv2.imshow('1. Original', img)
cv2.imshow('2. Gradient', mag)
cv2.imshow('3. Refined', refined)
cv2.imshow('4. Binary', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

# refined之后还要二值化的原因
# 1.
# refined的结果是梯度幅值（灰度图）
# refined保留的是局部最大值的原始梯度强度（0 - 255
# 之间的值）
#
# 不同边缘的强度不同：强边缘值大（200 +），弱边缘值小（30 - 50）
#
# 2.
# 二值化的作用
# 处理阶段
# 像素值范围
# 含义
# refined
# 0 - 255
# 边缘强度（越亮越强）
# binary
# 0
# 或255
# 是 / 否是边缘
# 3.
# 为什么要转成二值？
# ① 明确区分边缘和非边缘
#
# text
# refined: [0, 0, 120, 45, 200, 0, 30, 0]
#           ↑      ↑    ↑   ↑
# 不是
# 是
# 是
# 是（但强度不同）
#
# binary: [0, 0, 255, 0, 255, 0, 0, 0]
#          ↑          ↑
# 只保留真正强的边缘
# ② 去除弱边缘（噪声）
#
# 阈值30以下的是噪声或弱纹理
#
# 二值化时用阈值过滤：_, binary = cv2.threshold(refined, 30, 255, cv2.THRESH_BINARY)
#
# 只保留真正重要的边缘
#
# ③ 后续处理的需要
#
# 轮廓检测：cv2.findContours()
# 需要二值图像
#
# 特征提取：边缘跟踪、链码表示需要明确的是 / 否
#
# 形态学操作：膨胀、腐蚀、闭运算等都需要二值图
#
# 4.
# 对比示例
# text
# refined（细化后）：
# ████████████    ← 边缘有强弱变化
# ↓ 阈值30
#
# binary（二值化后）：
# ██████    ███    ← 只保留强边缘，弱边缘被过滤
# 5.
# 一句话总结
# refined找出了可能是边缘的位置（候选），二值化决定哪些真的是边缘（筛选）