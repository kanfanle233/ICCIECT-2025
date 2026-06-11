"""
教学示例：lbp

- 功能：演示 纹理分析 中与“lbp”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern


BASE_DIR = Path(__file__).resolve().parent

def local_path(name: str) -> str:
    """返回脚本目录下的资源路径，避免依赖当前工作目录。"""
    return str(BASE_DIR / name)
def process_lbp(image_path):
    """计算图像的LBP特征并显示归一化LBP图和直方图。"""
    # 1. 参数设置
    radius = 1          # LBP算法中范围半径的取值
    n_points = 8 * radius # 领域像素点数
    method = 'default'  # 使用默认LBP (0-255范围)

    # 2. 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    # 3. 计算lbp
    lbp = local_binary_pattern(img, n_points, radius, method)

    # 4. 归一化到0-255
    # LBP结果在default下已经是0-255，为了稳妥将其归一化到0-255范围，并转为uint8
    lbp_norm = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 5. 计算lbp直方图
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))

    # 6. 归一化直方图
    hist_norm = hist.astype("float") / (hist.sum() + 1e-7)

    # 7. 显示结果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title(f'Original: {image_path}')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(lbp_norm, cmap='gray')
    plt.title('LBP (Normalized 0-255)')
    plt.axis('off')

    plt.subplot(133)
    plt.bar(np.arange(256), hist_norm, width=1.0)
    plt.title('Normalized LBP Histogram')
    plt.xlim([0, 256])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_lbp(local_path("11.jpg"))
    process_lbp(local_path("22.jpg"))
