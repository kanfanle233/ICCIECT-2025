"""
教学示例：saliency hc

- 功能：演示 显著性检测 中与“saliency hc”相关的核心流程。
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


def calculate_hc_saliency(image_path=local_path("test.jpg")):
    """
    HC (Histogram Contrast): 基于直方图对比度的显著性检测
    原理: 像素显著性 = 该像素颜色与图像中所有其他颜色的距离加权之和
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取 {image_path}")
        return

    # 为了加速计算，将颜色量化到较少等级 (例如 64^3 或更少)
    # 这里使用简单的量化策略：将每个通道除以 8 (0-255 -> 0-31)
    quantize_factor = 8
    img_quant = (img // quantize_factor) * quantize_factor + quantize_factor // 2

    # 展平图像以便统计
    pixels = img_quant.reshape(-1, 3)

    # 计算颜色直方图 (频率)
    # 使用字典存储颜色及其出现次数，比多维数组更灵活
    color_counts = {}
    for p in pixels:
        key = tuple(p)
        color_counts[key] = color_counts.get(key, 0) + 1

    total_pixels = len(pixels)
    unique_colors = list(color_counts.keys())
    n_colors = len(unique_colors)

    # 预计算颜色之间的欧氏距离和频率权重
    # S(c_i) = sum( D(c_i, c_j) * F(c_j) ) for all j
    saliency_map = np.zeros(img.shape[:2], dtype=np.float32)

    # 将 unique_colors 转为 numpy 数组加速计算
    colors_arr = np.array(unique_colors, dtype=np.float32)
    counts_arr = np.array([color_counts[c] for c in unique_colors], dtype=np.float32)
    freqs_arr = counts_arr / total_pixels

    # 逐像素计算 (优化版：先计算每个唯一颜色的显著性，再映射回图像)
    color_saliency = np.zeros(n_colors, dtype=np.float32)

    # 计算距离矩阵可能很大，这里采用循环优化或分批处理
    # 对于演示代码，我们直接双重循环计算每个唯一颜色的显著性
    # 注意：如果颜色种类太多，这一步会慢。量化后通常在几千种以内。
    for i in range(n_colors):
        c_i = colors_arr[i]
        # 计算 c_i 到所有其他颜色的距离
        dists = np.linalg.norm(colors_arr - c_i, axis=1)
        # 加权求和: sum(dist * freq)
        color_saliency[i] = np.sum(dists * freqs_arr)

    # 将显著性值映射回图像
    # 创建查找表
    for idx, color in enumerate(unique_colors):
        # 找到图像中等于该颜色的所有位置
        # 这种方法在大图中较慢，改用重塑映射
        pass

    # 更高效的映射方法：
    # 1. 将量化后的图像转为线性索引或直接匹配
    # 这里使用简单的广播匹配 (对于大图可能稍慢，但逻辑清晰)
    # 为了速度，我们构建一个字典映射 color -> saliency_value
    saliency_lookup = {unique_colors[i]: color_saliency[i] for i in range(n_colors)}

    # 应用映射
    h, w = img.shape[:2]
    for y in range(h):
        for x in range(w):
            col = tuple(img_quant[y, x])
            saliency_map[y, x] = saliency_lookup[col]

    # 归一化到 0-255
    saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)
    saliency_uint8 = saliency_map.astype(np.uint8)

    # 可视化
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 4, 1)
    plt.title("1. 原始图像")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("2. 颜色量化后")
    plt.imshow(cv2.cvtColor(img_quant, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("3. HC 显著性图")
    plt.imshow(saliency_uint8, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("4. 二值化结果 (Otsu)")
    _, binary = cv2.threshold(saliency_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.imshow(binary, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    print("HC 算法执行完毕。")


if __name__ == "__main__":
    calculate_hc_saliency(local_path("test.jpg"))
