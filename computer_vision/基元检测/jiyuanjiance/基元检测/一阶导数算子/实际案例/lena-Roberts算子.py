# -*- coding: utf-8 -*-
"""
教学示例：lena-Roberts算子

- 功能：演示 一阶导数边缘检测 中与“lena-Roberts算子”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""
from pathlib import Path

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent

def local_path(name: str) -> str:
    """返回脚本目录下的资源路径，避免依赖当前工作目录。"""
    return str(BASE_DIR / name)
# 尝试多种后端
backends = ['TkAgg', 'Qt5Agg', 'Agg']
success = False

for backend in backends:
    try:
        matplotlib.use(backend, force=True)
        print(f"尝试使用后端: {backend}")
        # 测试创建一个简单的图
        plt.figure()
        plt.close()
        success = True
        break
    except:
        continue

if not success:
    print("警告：无法设置matplotlib后端，使用OpenCV显示")
    # 回退到OpenCV显示
    img = cv2.imread(local_path("lena.png"))
    if img is None:
        img = np.ones((256, 256, 3), dtype=np.uint8) * 128
        cv2.rectangle(img, (50, 50), (206, 206), (255, 255, 255), 2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernelx = np.array([[1, 0], [0, -1]], dtype=int)
    kernely = np.array([[0, 1], [-1, 0]], dtype=int)
    x = cv2.filter2D(gray, cv2.CV_16S, kernelx)
    y = cv2.filter2D(gray, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    cv2.imshow('Original', img)
    cv2.imshow('Roberts', Roberts)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

# 继续使用matplotlib显示
img = cv2.imread(local_path("lena.png"))
if img is None:
    print("使用测试图像...")
    img = np.ones((256, 256, 3), dtype=np.uint8) * 128
    cv2.rectangle(img, (50, 50), (206, 206), (255, 255, 255), 2)
    lenna_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
else:
    lenna_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Roberts算子
kernelx = np.array([[1, 0], [0, -1]], dtype=int)
kernely = np.array([[0, 1], [-1, 0]], dtype=int)

x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)

absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# 显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

titles = ['原始图像', 'Roberts算子']
images = [lenna_img, Roberts]

plt.figure(figsize=(10, 5))
for i in range(2):
    plt.subplot(1, 2, i + 1)
    if i == 0:
        plt.imshow(images[i])
    else:
        plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()