# # -*- coding: utf-8 -*-
"""
教学示例：边界闭合

- 功能：演示 基元检测 中与“边界闭合”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

def local_path(name: str) -> str:
    """返回脚本目录下的资源路径，避免依赖当前工作目录。"""
    return str(BASE_DIR / name)
# import cv2
# import numpy as np
#
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
#
# # 读取图像
# img = cv2.imread(local_path("lena.png"))
# if img is None:
#     img = np.zeros((300, 300, 3), dtype=np.uint8)
#     cv2.rectangle(img, (50, 50), (250, 250), 255, 2)
#     cv2.circle(img, (150, 150), 80, 255, 2)
#
# # 灰度图
# if len(img.shape) == 3:
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# else:
#     gray = img.copy()
#
# # (a) 原始梯度图 - 使用Sobel计算梯度幅值
# sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
# sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
# gradient = np.sqrt(sobel_x**2 + sobel_y**2)
# gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#
# # (b) 阈值处理后的梯度图
# _, threshold = cv2.threshold(gradient, 50, 255, cv2.THRESH_BINARY)
#
# # (c) 过零点检测（边缘闭合）
# laplacian = cv2.Laplacian(gray, cv2.CV_64F)
# zero_cross = np.zeros_like(gray)
# h, w = laplacian.shape
# for i in range(1, h-1):
#     for j in range(1, w-1):
#         if (laplacian[i, j] * laplacian[i-1, j] < 0) or \
#            (laplacian[i, j] * laplacian[i+1, j] < 0) or \
#            (laplacian[i, j] * laplacian[i, j-1] < 0) or \
#            (laplacian[i, j] * laplacian[i, j+1] < 0):
#             zero_cross[i, j] = 255
#
# # 显示三个图
# plt.figure(figsize=(12, 4))
#
# plt.subplot(1, 3, 1)
# plt.imshow(gradient, cmap='gray')
# plt.title('(a) 原始梯度图')
# plt.axis('off')
#
# plt.subplot(1, 3, 2)
# plt.imshow(threshold, cmap='gray')
# plt.title('(b) 阈值处理后')
# plt.axis('off')
#
# plt.subplot(1, 3, 3)
# plt.imshow(zero_cross, cmap='gray')
# plt.title('(c) 过零点检测')
# plt.axis('off')
#
# plt.tight_layout()
# plt.show()
#
#
#
#
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 读取图像
img = cv2.imread(local_path("lena.png"))
if img is None:
    img = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (250, 250), 255, 2)
    cv2.circle(img, (150, 150), 80, 255, 2)

# 灰度图
if len(img.shape) == 3:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    gray = img.copy()

# 确保图像是300x300
gray = cv2.resize(gray, (300, 300))

# (a) 原始梯度图
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
gradient = np.sqrt(sobel_x**2 + sobel_y**2)
gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# (b) 阈值处理
_, threshold = cv2.threshold(gradient, 50, 255, cv2.THRESH_BINARY)

# (c) 过零点检测
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
zero_cross = np.zeros_like(gray)
h, w = laplacian.shape
for i in range(1, h-1):
    for j in range(1, w-1):
        if (laplacian[i, j] * laplacian[i-1, j] < 0) or \
           (laplacian[i, j] * laplacian[i+1, j] < 0) or \
           (laplacian[i, j] * laplacian[i, j-1] < 0) or \
           (laplacian[i, j] * laplacian[i, j+1] < 0):
            zero_cross[i, j] = 255

# 创建画布
canvas = np.zeros((300, 900, 3), dtype=np.uint8)

# 转换并复制到画布
canvas[0:300, 0:300] = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
canvas[0:300, 300:600] = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
canvas[0:300, 600:900] = cv2.cvtColor(zero_cross, cv2.COLOR_GRAY2BGR)



# # 如果想要中文
# 方法1
# PIL添加中文
canvas_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(canvas_pil)
font = ImageFont.truetype(local_path("simhei.ttf"), 20)  # 使用黑体
draw.text((10, 10), "(a) 梯度图", font=font, fill=(0, 255, 0))
draw.text((310, 10), "(b) 阈值图", font=font, fill=(0, 255, 0))
draw.text((610, 10), "(c) 过零点", font=font, fill=(0, 255, 0))
# 显示
cv2.imshow('Edge Closing', cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_RGB2BGR))


# 方法2
# # 使用PIL添加中文
# canvas_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
# draw = ImageDraw.Draw(canvas_pil)
#

# # 加载中文字体（需要系统有中文字体）
# try:
#     font = ImageFont.truetype("simhei.ttf", 20)  # 黑体
# except:
#     try:
#         font = ImageFont.truetype("msyh.ttc", 20)  # 微软雅黑
#     except:
#         font = ImageFont.load_default()
# # 添加中文标签
# draw.text((10, 10), "(a) 梯度图", font=font, fill=(0, 255, 0))
# draw.text((310, 10), "(b) 阈值图", font=font, fill=(0, 255, 0))
# draw.text((610, 10), "(c) 过零点", font=font, fill=(0, 255, 0))
# # 转回OpenCV格式
# canvas = cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_RGB2BGR)


# 不要求中文
# # 添加标签
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(canvas, '(a) 梯度图', (10, 30), font, 0.7, (0, 255, 0), 2)
# cv2.putText(canvas, '(b) 阈值图', (310, 30), font, 0.7, (0, 255, 0), 2)
# cv2.putText(canvas, '(c) 过零点', (610, 30), font, 0.7, (0, 255, 0), 2)
#
# # 显示
# cv2.imshow('Edge Closing', canvas)

cv2.waitKey(0)
cv2.destroyAllWindows()