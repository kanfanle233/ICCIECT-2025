# -*- coding: utf-8 -*-
"""
教学示例：拉普拉斯 马尔 canny自带的

- 功能：演示 二阶导数与边缘检测 中与“拉普拉斯 马尔 canny自带的”相关的核心流程。
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

# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. 拉普拉斯算子
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

# 2. 马尔算子 (LoG)
blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
log = cv2.Laplacian(blurred, cv2.CV_64F)
log = np.uint8(np.absolute(log))

# 3. Canny算子
canny = cv2.Canny(gray, 50, 150)

# 创建画布显示四张图
canvas = np.zeros((600, 800, 3), dtype=np.uint8)

# 调整图像大小
img_small = cv2.resize(img, (400, 300))
lap_small = cv2.resize(laplacian, (400, 300))
log_small = cv2.resize(log, (400, 300))
canny_small = cv2.resize(canny, (400, 300))

# 将灰度图转BGR以便显示
lap_small = cv2.cvtColor(lap_small, cv2.COLOR_GRAY2BGR)
log_small = cv2.cvtColor(log_small, cv2.COLOR_GRAY2BGR)
canny_small = cv2.cvtColor(canny_small, cv2.COLOR_GRAY2BGR)

# 放置图像到画布
canvas[0:300, 0:400] = img_small
canvas[0:300, 400:800] = lap_small
canvas[300:600, 0:400] = log_small
canvas[300:600, 400:800] = canny_small

# 添加文字标签
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(canvas, 'Original', (10, 30), font, 1, (0, 255, 0), 2)
cv2.putText(canvas, 'Laplacian', (410, 30), font, 1, (0, 255, 0), 2)
cv2.putText(canvas, 'LoG', (10, 330), font, 1, (0, 255, 0), 2)
cv2.putText(canvas, 'Canny', (410, 330), font, 1, (0, 255, 0), 2)

# 显示
cv2.imshow('Edge Detection Comparison', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()



# # 有注释的代码
# # -*- coding: utf-8 -*-  # 设置文件编码为UTF-8，支持中文
# import cv2  # 导入OpenCV库，用于图像处理
# import numpy as np  # 导入NumPy库，用于数组操作
#
# # 读取图像
# # cv2.imread() 读取图像文件，参数为文件路径
# # 返回：成功则返回图像数组，失败返回None
# img = cv2.imread(local_path("lena.png"))
#
# # 如果图像读取失败（img为None），则创建测试图像
# if img is None:
#     # np.zeros() 创建指定大小的零数组（黑色图像）
#     # 参数：(300, 300, 3) 表示高度300像素，宽度300像素，3通道(BGR)
#     # dtype=np.uint8 表示数据类型为8位无符号整数(0-255)
#     img = np.zeros((300, 300, 3), dtype=np.uint8)
#
#     # cv2.rectangle() 绘制矩形
#     # 参数：图像, 左上角坐标(50,50), 右下角坐标(250,250), 颜色(255,255,255)白色, 线宽2
#     cv2.rectangle(img, (50, 50), (250, 250), (255, 255, 255), 2)
#
#     # cv2.circle() 绘制圆形
#     # 参数：图像, 圆心坐标(150,150), 半径80, 颜色白色, 线宽2
#     cv2.circle(img, (150, 150), 80, (255, 255, 255), 2)
#
# # 转换为灰度图
# # cv2.cvtColor() 颜色空间转换函数
# # 参数：输入图像, 转换代码 cv2.COLOR_BGR2GRAY (BGR转灰度)
# # 返回：单通道灰度图像
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # 1. 拉普拉斯算子
# # cv2.Laplacian() 拉普拉斯边缘检测函数
# # 参数1: gray - 输入灰度图像
# # 参数2: cv2.CV_64F - 输出图像深度，64位浮点数（防止数据溢出）
# # 返回：拉普拉斯变换结果（包含正负值）
# laplacian = cv2.Laplacian(gray, cv2.CV_64F)
#
# # np.absolute() 计算绝对值，将负值转为正值（边缘强度）
# # np.uint8() 将浮点数转换为8位无符号整数(0-255)
# laplacian = np.uint8(np.absolute(laplacian))
#
# # 2. 马尔算子 (LoG = Laplacian of Gaussian)
# # cv2.GaussianBlur() 高斯滤波（平滑图像，减少噪声）
# # 参数1: gray - 输入图像
# # 参数2: (5, 5) - 高斯核大小，5x5的窗口
# # 参数3: 1.4 - 标准差sigma，控制平滑程度
# blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
#
# # 对平滑后的图像应用拉普拉斯算子
# log = cv2.Laplacian(blurred, cv2.CV_64F)
# # 取绝对值并转换为8位无符号整数
# log = np.uint8(np.absolute(log))
#
# # 3. Canny算子
# # cv2.Canny() Canny边缘检测函数（最优边缘检测算法）
# # 参数1: gray - 输入灰度图像
# # 参数2: 50 - 低阈值，低于此值的点不被认为是边缘
# # 参数3: 150 - 高阈值，高于此值的点被确定为强边缘
# # 返回：二值边缘图像（边缘点为白色255，背景为黑色0）
# canny = cv2.Canny(gray, 50, 150)
#
# # 创建画布显示四张图
# # np.zeros() 创建黑色画布
# # 参数：(600, 800, 3) - 高度600像素，宽度800像素，3通道彩色
# canvas = np.zeros((600, 800, 3), dtype=np.uint8)
#
# # 调整图像大小以便在画布上显示
# # cv2.resize() 图像缩放函数
# # 参数1: 输入图像
# # 参数2: (400, 300) - 目标尺寸（宽度400，高度300）
# img_small = cv2.resize(img, (400, 300))
# lap_small = cv2.resize(laplacian, (400, 300))
# log_small = cv2.resize(log, (400, 300))
# canny_small = cv2.resize(canny, (400, 300))
#
# # 将灰度图转换为BGR格式（因为画布是3通道彩色）
# # cv2.cvtColor() 颜色空间转换
# # 参数：灰度图像, cv2.COLOR_GRAY2BGR - 灰度转BGR（单通道复制到三个通道）
# lap_small = cv2.cvtColor(lap_small, cv2.COLOR_GRAY2BGR)
# log_small = cv2.cvtColor(log_small, cv2.COLOR_GRAY2BGR)
# canny_small = cv2.cvtColor(canny_small, cv2.COLOR_GRAY2BGR)
#
# # 放置图像到画布的指定位置
# # 画布索引 [y起始:y结束, x起始:x结束]
# canvas[0:300, 0:400] = img_small  # 左上角：原图
# canvas[0:300, 400:800] = lap_small  # 右上角：拉普拉斯
# canvas[300:600, 0:400] = log_small  # 左下角：LoG
# canvas[300:600, 400:800] = canny_small  # 右下角：Canny
#
# # 添加文字标签
# # cv2.FONT_HERSHEY_SIMPLEX - 字体类型（简单仿宋体）
# font = cv2.FONT_HERSHEY_SIMPLEX
#
# # cv2.putText() 在图像上添加文字
# # 参数1: canvas - 目标图像
# # 参数2: 要添加的文字内容
# # 参数3: (10, 30) - 文字左下角坐标(x, y)
# # 参数4: font - 字体类型
# # 参数5: 1 - 字体大小
# # 参数6: (0, 255, 0) - 颜色（BGR格式，这里是绿色）
# # 参数7: 2 - 线宽
# cv2.putText(canvas, 'Original', (10, 30), font, 1, (0, 255, 0), 2)
# cv2.putText(canvas, 'Laplacian', (410, 30), font, 1, (0, 255, 0), 2)
# cv2.putText(canvas, 'LoG', (10, 330), font, 1, (0, 255, 0), 2)
# cv2.putText(canvas, 'Canny', (410, 330), font, 1, (0, 255, 0), 2)
#
# # 显示图像
# # cv2.imshow() 创建窗口显示图像
# # 参数1: 'Edge Detection Comparison' - 窗口标题
# # 参数2: canvas - 要显示的图像
# cv2.imshow('Edge Detection Comparison', canvas)
#
# # cv2.waitKey(0) 等待按键
# # 参数0表示无限等待，直到用户按键
# # 返回值：按下的键的ASCII码
# cv2.waitKey(0)
#
# # cv2.destroyAllWindows() 关闭所有OpenCV创建的窗口
# # 释放资源
# cv2.destroyAllWindows()