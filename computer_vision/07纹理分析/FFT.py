"""
教学示例：FFT

- 功能：演示 纹理分析 中与“FFT”相关的核心流程。
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
def process_fft(image_path):
    """读取灰度图像并计算二维FFT，显示原图与频谱幅值对比。"""
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    # 计算2D FFT
    f = np.fft.fft2(img)
    # 将低频分量移动到中心
    fshift = np.fft.fftshift(f)
    
    # 计算频谱幅值（窗口化结果展示）
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # 给出窗口化结果进行显示，不保存结果图片
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title(f'Original Image: {image_path}')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('FFT Magnitude Spectrum')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_fft(local_path("11.jpg"))
    process_fft(local_path("22.jpg"))
