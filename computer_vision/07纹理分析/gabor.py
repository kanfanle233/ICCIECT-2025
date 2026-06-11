"""
教学示例：gabor

- 功能：演示 纹理分析 中与“gabor”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent

def local_path(name: str) -> str:
    """返回脚本目录下的资源路径，避免依赖当前工作目录。"""
    return str(BASE_DIR / name)
def process_gabor(image_path):
    """使用Gabor滤波器对图像进行纹理特征提取，显示原图、核和滤波结果。"""
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    # 设置Gabor滤波器参数
    ksize = 31 # 滤波器核大小
    sigma = 4.0 # 高斯包络的标准差
    theta = np.pi / 4 # Gabor条纹方向 (这里设为45度)
    lamda = 10.0 # 正弦因子的波长
    gamma = 0.5 # 空间高宽比
    phi = 0 # 相位偏移
    
    # 创建Gabor核
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
    
    # 使用Gabor核对图像进行滤波
    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, kernel)
    
    # 显示结果窗体，不保存文件
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title(f'Original Image: {image_path}')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(kernel, cmap='gray')
    plt.title('Gabor Kernel (45 degree)')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(filtered_img, cmap='gray')
    plt.title('Filtered Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_gabor(local_path("11.jpg"))
    process_gabor(local_path("22.jpg"))
