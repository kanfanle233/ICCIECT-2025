"""
教学示例：saliency ft

- 功能：演示 显著性检测 中与“saliency ft”相关的核心流程。
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
# 1. 设置字体为 SimHei (黑体)，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
# 2. 解决保存图像时负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False


def calculate_ft_saliency(image_path=local_path("test.jpg")):
    """
    FT (Frequency Tuned): 频率调谐显著性检测
    原理: 计算每个像素颜色与图像全局平均颜色的欧氏距离 (在 Lab 空间)
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取 {image_path}")
        return

    # 转换到 Lab 色彩空间 (感知均匀)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 计算全局平均颜色 (忽略 L 通道的亮度影响，或全部使用，经典FT使用全部)
    # 这里使用 a 和 b 通道通常效果更好，但标准FT使用 L,a,b
    mean_val = np.mean(img_lab, axis=(0, 1))

    # 计算每个像素与均值的距离
    # |I(x,y) - Mean(I)|
    diff = img_lab - mean_val

    # 欧氏距离平方和 (或者开根号，归一化后效果类似)
    # 只使用 a, b 通道往往能更好抑制光照变化，这里演示标准全通道
    saliency_map = np.sum(diff ** 2, axis=2)

    # 归一化到 0-255
    saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)
    saliency_uint8 = saliency_map.astype(np.uint8)

    # 可选：高斯模糊平滑
    saliency_blur = cv2.GaussianBlur(saliency_uint8, (5, 5), 0)

    # 可视化
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 5, 1)
    plt.title("1. 原始图像")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.title("2. Lab 均值颜色块")
    mean_color_img = np.ones((50, 50, 3), dtype=np.float32) * mean_val
    mean_color_bgr = cv2.cvtColor(mean_color_img.astype(np.uint8), cv2.COLOR_LAB2BGR)
    plt.imshow(cv2.cvtColor(mean_color_bgr, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.title("3. FT 显著性图 (原始)")
    plt.imshow(saliency_uint8, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.title("4. FT 显著性图 (平滑)")
    plt.imshow(saliency_blur, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.title("5. 二值化结果 (Otsu)")
    _, binary = cv2.threshold(saliency_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.imshow(binary, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    print("FT 算法执行完毕。")


if __name__ == "__main__":
    calculate_ft_saliency(local_path("test.jpg"))
