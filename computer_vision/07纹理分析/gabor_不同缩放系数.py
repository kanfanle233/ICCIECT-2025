"""
教学示例：gabor 不同缩放系数

- 功能：演示 纹理分析 中与“gabor 不同缩放系数”相关的核心流程。
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
def process_gabor_with_scales(image_path):
    """使用不同缩放系数的Gabor滤波器对图像进行多尺度纹理特征对比。"""
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    # 基础Gabor参数（风格参考原脚本）
    ksize = 31  # 滤波器核大小
    base_sigma = 4.0  # 高斯包络的基础标准差
    theta = np.pi / 4  # Gabor条纹方向（45度）
    base_lamda = 10.0  # 正弦因子的基础波长
    gamma = 0.5  # 空间高宽比
    phi = 0  # 相位偏移

    # 不同缩放系数：同时缩放sigma和lamda
    scale_factors = [0.5, 1.0, 1.5, 2.0]

    kernels = []
    filtered_images = []
    for scale in scale_factors:
        sigma = base_sigma * scale
        lamda = base_lamda * scale

        # 创建对应缩放系数的Gabor核
        kernel = cv2.getGaborKernel(
            (ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F
        )
        kernels.append(kernel)

        # 使用Gabor核滤波
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, kernel)
        filtered_images.append(filtered_img)

    # 对比展示：原图 + 不同缩放系数下的核和滤波结果
    fig, axes = plt.subplots(2, len(scale_factors) + 1, figsize=(18, 8))
    fig.suptitle(f"Gabor不同缩放系数对比: {image_path}", fontsize=14)

    # 第一列放原图（两行都显示，便于与核和结果直接对照）
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[1, 0].imshow(img, cmap="gray")
    axes[1, 0].set_title("Original Image")
    axes[1, 0].axis("off")

    # 其余列显示不同缩放系数下的核与滤波结果
    for i, scale in enumerate(scale_factors):
        axes[0, i + 1].imshow(kernels[i], cmap="gray")
        axes[0, i + 1].set_title(f"Kernel x{scale}")
        axes[0, i + 1].axis("off")

        axes[1, i + 1].imshow(filtered_images[i], cmap="gray")
        axes[1, i + 1].set_title(f"Filtered x{scale}")
        axes[1, i + 1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    process_gabor_with_scales(local_path("11.jpg"))
    process_gabor_with_scales(local_path("22.jpg"))
