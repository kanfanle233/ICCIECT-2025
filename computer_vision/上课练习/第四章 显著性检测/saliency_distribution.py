"""
教学示例：saliency distribution

- 功能：演示 显著性检测 中与“saliency distribution”相关的核心流程。
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


def calculate_distribution_based_saliency(image_path):
    """
    基于对比度分布的显著性检测 (Frequency Tuned - FT Algorithm)
    原理: S(x) = || I(x) - Mean(I) ||
    即：像素显著性 = 该像素颜色与图像全局平均颜色的距离
    """
    # 1. 读取图像
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError("无法读取图像，请检查路径")

    # 转换为 Lab 颜色空间 (Lab 空间中欧氏距离更符合人眼感知差异)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 分离 L, a, b 通道
    L, a, b = cv2.split(img_lab)

    # 2. 计算全局颜色分布的中心 (均值)
    # 这里只计算 a 和 b 通道的均值，因为 L (亮度) 通道对显著性贡献较小且易受光照影响
    # 原论文建议忽略 L 通道，仅使用色度通道 (a, b) 计算对比度
    mean_a = np.mean(a)
    mean_b = np.mean(b)

    print(f"全局颜色分布中心 -> a: {mean_a:.2f}, b: {mean_b:.2f}")

    # 3. 计算每个像素与全局均值的距离 (欧氏距离)
    # 公式: S = sqrt( (a - mean_a)^2 + (b - mean_b)^2 )
    # 使用向量化操作加速计算
    dist_a = a - mean_a
    dist_b = b - mean_b

    saliency_map = np.sqrt(dist_a ** 2 + dist_b ** 2)

    # 4. 归一化到 0-255
    # 注意：FT 算法生成的图通常不需要额外的平滑处理，因为它天然保留了高频细节（边缘）
    # 但为了视觉效果和阈值分割，通常会做一个轻微的高斯模糊
    saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)
    saliency_map = saliency_map.astype(np.uint8)

    # 可选：轻微高斯模糊以连接断裂的边缘 (FT 原论文中有时不做，但实际应用中做一点效果更好)
    saliency_blur = cv2.GaussianBlur(saliency_map, (3, 3), 0)

    # 5. 二值化掩膜 (使用 Otsu 自动阈值)
    _, binary_mask = cv2.threshold(saliency_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 为了演示“分布”的概念，我们还可以计算并返回直方图数据
    hist_a = cv2.calcHist([a], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])

    return img_bgr, saliency_blur, binary_mask, (mean_a, mean_b), (hist_a, hist_b)


# --- 主程序 ---
if __name__ == "__main__":
    image_file = local_path("test.jpg")

    import os

    # 如果没有测试图，生成一张具有明显颜色分布特征的图
    if not os.path.exists(image_file):
        print(f"未找到 {image_file}，正在生成分布测试图像...")
        # 创建一个背景占主导，前景独特的图像
        h, w = 400, 400
        test_img = np.zeros((h, w, 3), dtype=np.uint8)

        # 背景：大面积的绿色 (Lab: L~50, a~-40, b~20 左右，这里用 BGR 近似)
        # BGR: (0, 150, 0) -> 绿色
        test_img[:] = [0, 150, 0]

        # 添加一些背景噪声，模拟真实分布
        noise = np.random.randint(-10, 10, (h, w, 3), dtype=np.int16)
        test_img = np.clip(test_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 前景：一个红色的圆 (与绿色分布差异大)
        # BGR: (0, 0, 255) -> 红色
        cv2.circle(test_img, (200, 200), 70, (0, 0, 255), -1)

        # 前景：一个黄色的方块
        # BGR: (0, 255, 255) -> 黄色
        cv2.rectangle(test_img, (50, 50), (120, 120), (0, 255, 255), -1)

        cv2.imwrite(image_file, test_img)
        print("测试图像已生成。")

    try:
        original, saliency, mask, global_mean, histograms = calculate_distribution_based_saliency(image_file)
        mean_a, mean_b = global_mean
        hist_a, hist_b = histograms

        plt.figure(figsize=(18, 4))

        # 图 1: 原始图像
        plt.subplot(1, 6, 1)
        plt.title("1. 原始图像")
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        # 图 2: 显著性热力图 (灰度)
        plt.subplot(1, 6, 2)
        plt.title("2. FT 显著性图\n(基于分布距离)")
        plt.imshow(saliency, cmap='gray')
        plt.axis('off')

        # 图 3: 伪彩色显著图
        plt.subplot(1, 6, 3)
        plt.title("3. 伪彩色映射")
        plt.imshow(saliency, cmap='jet')
        plt.axis('off')

        # 图 4: 二值化结果
        plt.subplot(1, 6, 4)
        plt.title("4. 提取掩膜")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        # 图 5: 'a' 通道颜色分布直方图
        plt.subplot(1, 6, 5)
        plt.title(f"5. 'a' 通道分布\n均值: {mean_a:.1f}")
        plt.plot(hist_a, color='green', linewidth=2)
        plt.axvline(x=mean_a, color='red', linestyle='--', label='Global Mean')
        plt.legend()
        plt.xlim(0, 256)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.3)

        # 图 6: 'b' 通道颜色分布直方图
        plt.subplot(1, 6, 6)
        plt.title(f"6. 'b' 通道分布\n均值: {mean_b:.1f}")
        plt.plot(hist_b, color='blue', linewidth=2)
        plt.axvline(x=mean_b, color='red', linestyle='--', label='Global Mean')
        plt.legend()
        plt.xlim(0, 256)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.show()

        print("处理完成！观察直方图可以看到：背景颜色聚集在均值附近，前景颜色远离均值。")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback

        traceback.print_exc()