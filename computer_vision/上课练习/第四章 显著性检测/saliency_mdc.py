"""
教学示例：saliency mdc

- 功能：演示 显著性检测 中与“saliency mdc”相关的核心流程。
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


def calculate_mdc_saliency(image_path=local_path("test.jpg"), kernel_size=9):
    """
    MDC (Minimum Directional Contrast): 最小方向对比度显著性检测
    原理: 计算4个方向的局部对比度，取最小值以抑制单向纹理背景
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取 {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    # 定义4个方向的线性结构元素用于计算邻域均值
    # 0: 水平, 1: 垂直, 2: 45度, 3: 135度
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size)),
        None,
        None
    ]

    # 构建对角线核
    k_diag = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    np.fill_diagonal(k_diag, 1)
    kernels[2] = k_diag

    k_anti = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    np.fill_diagonal(np.fliplr(k_anti), 1)
    kernels[3] = k_anti

    # 计算中心区域的均值 (用小核平滑代表中心)
    center_mean = cv2.blur(gray, (3, 3))

    contrasts = []

    for k in kernels:
        # 计算该方向邻域的均值
        # 注意归一化核
        k_norm = k / np.sum(k)
        neighbor_mean = cv2.filter2D(gray, -1, k_norm)

        # 对比度 = |中心 - 邻域均值|
        diff = cv2.absdiff(center_mean, neighbor_mean)
        contrasts.append(diff)

    # 堆叠并取最小值
    contrasts_stack = np.stack(contrasts, axis=2)
    mdc_map = np.min(contrasts_stack, axis=2)

    # 归一化
    mdc_map = cv2.normalize(mdc_map, None, 0, 255, cv2.NORM_MINMAX)
    mdc_uint8 = mdc_map.astype(np.uint8)

    # 后处理：高斯模糊
    mdc_blur = cv2.GaussianBlur(mdc_uint8, (7, 7), 0)

    # 可视化
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 6, 1)
    plt.title("1. 原始图像")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 6, 2)
    plt.title("2. 水平方向对比度")
    plt.imshow(contrasts[0], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 6, 3)
    plt.title("3. 垂直方向对比度")
    plt.imshow(contrasts[1], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 6, 4)
    plt.title("4. 对角方向对比度")
    plt.imshow(contrasts[2], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 6, 5)
    plt.title("5. MDC (最小值)\n纹理被抑制")
    plt.imshow(mdc_uint8, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 6, 6)
    plt.title("6. 最终结果 (平滑)")
    plt.imshow(mdc_blur, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    print("MDC 算法执行完毕。")


if __name__ == "__main__":
    calculate_mdc_saliency(local_path("test.jpg"))
