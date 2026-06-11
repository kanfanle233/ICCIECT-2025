# -*- coding: utf-8 -*-
"""
教学示例：lena-3个算子

- 功能：演示 一阶导数边缘检测 中与“lena-3个算子”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""
from pathlib import Path

import cv2
import numpy as np
import matplotlib


BASE_DIR = Path(__file__).resolve().parent

def local_path(name: str) -> str:
    """返回脚本目录下的资源路径，避免依赖当前工作目录。"""
    return str(BASE_DIR / name)
matplotlib.use('TkAgg')  # 设置后端，避免PyCharm兼容性问题
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def roberts_operator(image):
    """
    Roberts算子边缘检测
    参数: image - 输入图像
    返回: Roberts边缘检测结果
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Roberts算子的两个核（2x2）
    # 45°方向核：检测从左上到右下的边缘
    kernel_45 = np.array([[1, 0],
                          [0, -1]], dtype=np.float32)
    # 135°方向核：检测从右上到左下的边缘
    kernel_135 = np.array([[0, 1],
                           [-1, 0]], dtype=np.float32)

    # 应用卷积
    grad_45 = cv2.filter2D(gray, cv2.CV_32F, kernel_45)
    grad_135 = cv2.filter2D(gray, cv2.CV_32F, kernel_135)

    # 计算梯度幅值
    magnitude = np.sqrt(grad_45 ** 2 + grad_135 ** 2)

    # 归一化到0-255
    result = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return result


def prewitt_operator(image):
    """
    Prewitt算子边缘检测
    参数: image - 输入图像
    返回: Prewitt边缘检测结果
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Prewitt算子的两个核（3x3）
    # 水平方向核：检测垂直边缘
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]], dtype=np.float32)
    # 垂直方向核：检测水平边缘
    kernel_y = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]], dtype=np.float32)

    # 应用卷积
    grad_x = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
    grad_y = cv2.filter2D(gray, cv2.CV_32F, kernel_y)

    # 计算梯度幅值
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 归一化到0-255
    result = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return result


def sobel_operator(image):
    """
    Sobel算子边缘检测
    参数: image - 输入图像
    返回: Sobel边缘检测结果
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Sobel算子的两个核（3x3，中间行/列权重更大）
    # 水平方向核：检测垂直边缘
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float32)
    # 垂直方向核：检测水平边缘
    kernel_y = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]], dtype=np.float32)

    # 应用卷积
    grad_x = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
    grad_y = cv2.filter2D(gray, cv2.CV_32F, kernel_y)

    # 计算梯度幅值
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 归一化到0-255
    result = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return result


def create_test_image():
    """创建一个测试图像（包含各种几何形状）"""

    # 创建黑色背景
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    # 添加一个白色矩形
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)

    # 添加一个白色圆形
    cv2.circle(img, (300, 100), 50, (255, 255, 255), -1)

    # 添加一条白色直线（水平）
    cv2.line(img, (50, 250), (350, 250), (255, 255, 255), 3)

    # 添加一条白色直线（垂直）
    cv2.line(img, (200, 50), (200, 350), (255, 255, 255), 3)

    # 添加一条白色对角线（45度）
    cv2.line(img, (50, 300), (350, 350), (255, 255, 255), 3)

    # 添加一条白色对角线（135度）
    cv2.line(img, (300, 300), (350, 350), (255, 255, 255), 3)

    # 添加一个三角形
    pts = np.array([[250, 300], [300, 350], [200, 350]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], (255, 255, 255))

    # 添加一些文字
    cv2.putText(img, 'TEST', (150, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return img


def main():
    """主函数"""

    print("=" * 60)
    print("边缘检测算子对比实验")
    print("=" * 60)

    # 读取图像（如果存在lena.png，否则使用测试图像）
    img = cv2.imread(local_path("lena.png"))

    if img is None:
        print("未找到lena.png，使用测试图像...")
        img = create_test_image()
        image_name = "测试图像"
    else:
        image_name = "Lena图像"

    # 转换为RGB用于显示
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 应用三个算子
    print("\n正在计算Roberts算子...")
    roberts_result = roberts_operator(img)

    print("正在计算Prewitt算子...")
    prewitt_result = prewitt_operator(img)

    print("正在计算Sobel算子...")
    sobel_result = sobel_operator(img)

    # 打印统计信息
    print("\n" + "=" * 60)
    print("结果统计:")
    print(f"Roberts: 范围 [{roberts_result.min()}, {roberts_result.max()}], 均值 {roberts_result.mean():.1f}")
    print(f"Prewitt: 范围 [{prewitt_result.min()}, {prewitt_result.max()}], 均值 {prewitt_result.mean():.1f}")
    print(f"Sobel:   范围 [{sobel_result.min()}, {sobel_result.max()}], 均值 {sobel_result.mean():.1f}")

    # 创建4张图的对比显示
    plt.figure(figsize=(16, 8))

    # 1. 原图
    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title(f'1. 原始图像 ({image_name})', fontsize=14)
    plt.axis('off')

    # 2. Roberts结果
    plt.subplot(2, 2, 2)
    plt.imshow(roberts_result, cmap='gray')
    plt.title('2. Roberts算子\n(2x2核，检测45°/135°边缘)', fontsize=14)
    plt.axis('off')

    # 3. Prewitt结果
    plt.subplot(2, 2, 3)
    plt.imshow(prewitt_result, cmap='gray')
    plt.title('3. Prewitt算子\n(3x3核，平均平滑)', fontsize=14)
    plt.axis('off')

    # 4. Sobel结果
    plt.subplot(2, 2, 4)
    plt.imshow(sobel_result, cmap='gray')
    plt.title('4. Sobel算子\n(3x3核，加权平滑)', fontsize=14)
    plt.axis('off')

    plt.tight_layout()
    plt.suptitle('边缘检测算子对比', fontsize=16, y=1.02)
    plt.show()

    # 显示三个算子的详细对比信息
    print("\n" + "=" * 60)
    print("算子特性对比:")
    print("-" * 60)
    print("Roberts: 2x2核 | 定位最准 | 对噪声敏感 | 计算最快 | 擅长对角线边缘")
    print("Prewitt: 3x3核 | 定位中等 | 抗噪中等   | 计算中等 | 各方向均匀")
    print("Sobel:   3x3核 | 定位中等 | 抗噪最好   | 计算中等 | 各向同性最好")
    print("=" * 60)

    # 可选：保存结果
    save_results = input("\n是否保存结果图像？(y/n): ")
    if save_results.lower() == 'y':
        cv2.imwrite(local_path("original.jpg"), img)
        cv2.imwrite(local_path("roberts_result.jpg"), roberts_result)
        cv2.imwrite(local_path("prewitt_result.jpg"), prewitt_result)
        cv2.imwrite(local_path("sobel_result.jpg"), sobel_result)
        print("结果已保存！")


if __name__ == "__main__":
    main()