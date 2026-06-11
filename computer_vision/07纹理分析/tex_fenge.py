"""
教学示例：tex fenge

- 功能：演示 纹理分析 中与“tex fenge”相关的核心流程。
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
def process_tex_fenge(image_path):
    """对图像进行纹理分割（阈值法），显示原图、分割结果和直方图。"""
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    # 设置纹理分割参数
    # 这里使用简单的阈值分割作为示例，您可以根据需要替换为更复杂的纹理分割算法
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # 显示结果窗体，不保存文件
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title(f'Original Image: {image_path}')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(binary_img, cmap='gray')
    plt.title('Texture Segmentation Result')
    plt.axis('off')
    
    plt.subplot(133)
    plt.hist(img.ravel(), 256, [0, 256])
    plt.title('Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_tex_fenge(local_path("11.jpg"))
    process_tex_fenge(local_path("22.jpg"))