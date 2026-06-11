"""
教学示例：saliency contrast HC

- 功能：演示 显著性检测 中与“saliency contrast HC”相关的核心流程。
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
from collections import Counter


def calculate_global_contrast_saliency(image_path):
    """
    基于直方图对比度 (HC) 的显著性检测简化实现
    原理：像素的显著性 = 该像素颜色与图像中所有其他像素颜色的距离之和
    """
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像，请检查路径")

    # 转换为 RGB (Matplotlib 需要) 和 Lab 颜色空间 (Lab 空间更符合人眼感知)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    h, w, _ = img_lab.shape
    pixels = img_lab.reshape(-1, 3)

    # 2. 颜色量化 (为了加速计算，将颜色聚类或减少颜色数量)
    # 这里使用简单的四舍五入量化，将颜色空间压缩，减少计算量
    # 实际工程中常用 K-Means 聚类
    quantize_factor = 10
    pixels_quantized = np.floor(pixels / quantize_factor) * quantize_factor

    # 统计每种颜色的出现频率
    # 将 float 数组转换为 tuple 以便哈希计数
    unique_colors, counts = np.unique(pixels_quantized, axis=0, return_counts=True)
    color_freq_map = {tuple(c): freq for c, freq in zip(unique_colors, counts)}

    total_pixels = h * w
    saliency_map_flat = np.zeros(pixels.shape[0])

    # 3. 计算显著性 (简化版：计算当前颜色与其他所有颜色的加权距离)
    # 优化：不遍历所有像素，而是遍历“独特颜色”
    unique_colors_list = list(color_freq_map.keys())

    print(f"正在计算显著性... 独特颜色数量: {len(unique_colors_list)}")

    # 预计算所有独特颜色之间的显著性值
    color_saliency = {}
    for i, c1 in enumerate(unique_colors_list):
        s_val = 0.0
        c1_vec = np.array(c1)
        for j, c2 in enumerate(unique_colors_list):
            if i == j:
                continue
            c2_vec = np.array(c2)
            # 欧氏距离
            dist = np.linalg.norm(c1_vec - c2_vec)
            # 显著性贡献 = 距离 * 该颜色的频率 (频率越高，对比越明显)
            s_val += dist * color_freq_map[c2]

        # 归一化 (可选，防止数值过大)
        color_saliency[c1] = s_val

    # 4. 映射回原图像素
    for idx, p in enumerate(pixels_quantized):
        key = tuple(p)
        saliency_map_flat[idx] = color_saliency[key]

    # 5. 重构显著图并归一化到 0-255
    saliency_map = saliency_map_flat.reshape(h, w)
    saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)
    saliency_map = saliency_map.astype(np.uint8)

    # 6. 后处理：高斯模糊 (去除噪点，使区域更连贯)
    saliency_map_blur = cv2.GaussianBlur(saliency_map, (5, 5), 0)

    # 7. 二值化 (获取显著区域掩膜)
    _, binary_mask = cv2.threshold(saliency_map_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img_rgb, saliency_map_blur, binary_mask


# --- 主程序 ---
if __name__ == "__main__":
    # 请使用您本地的图片路径，或者下载一张测试图
    # 这里假设当前目录下有一张 test.jpg
    image_file = local_path("test.jpg")

    try:
        # 如果没有测试图，创建一个简单的合成图用于演示
        import os

        if not os.path.exists(image_file):
            print(f"未找到 {image_file}，正在生成测试图像...")
            test_img = np.zeros((400, 400, 3), dtype=np.uint8)
            test_img[:] = [200, 200, 200]  # 灰色背景
            cv2.circle(test_img, (200, 200), 80, (0, 0, 255), -1)  # 红色圆 (BGR)
            cv2.rectangle(test_img, (50, 50), (150, 150), (255, 255, 0), -1)  # 青色方块
            cv2.imwrite(image_file, test_img)
            print("测试图像已生成。")

        original, saliency, mask = calculate_global_contrast_saliency(image_file)

        # 显示结果
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 4, 1)
        plt.title("1. 原始图像")
        plt.imshow(original)
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.title("2. 显著性热力图 (灰度)")
        plt.imshow(saliency, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.title("3. 显著性热力图 (彩色映射)")
        plt.imshow(saliency, cmap='jet')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.title("4. 二值化掩膜 (提取区域)")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"发生错误: {e}")