"""
教学示例：saliency mser

- 功能：演示 显著性检测 中与“saliency mser”相关的核心流程。
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


def calculate_mser_saliency(image_path=local_path("test.jpg")):
    """
    MSER (Maximally Stable Extremal Regions): 最稳定区域检测
    原理: 寻找在灰度阈值变化过程中面积保持稳定的连通区域
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取 {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 创建 MSER 检测器 (使用默认参数)
    mser = cv2.MSER_create()

    # 检测区域
    regions, bboxes = mser.detectRegions(gray)

    # 生成掩膜
    mask = np.zeros((h, w), dtype=np.uint8)

    # 填充区域 (使用凸包以保证区域完整性)
    for region in regions:
        hull = cv2.convexHull(region)
        cv2.drawContours(mask, [hull], -1, 255, -1)

    # 生成热力图风格的显著性图 (模糊掩膜)
    saliency_float = mask.astype(np.float32) / 255.0
    saliency_blur = cv2.GaussianBlur(saliency_float, (9, 9), 0)
    saliency_uint8 = (saliency_blur * 255).astype(np.uint8)

    # 形态学清理 (可选)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 可视化
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 6, 1)
    plt.title("1. 原始图像")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 6, 2)
    plt.title(f"2. MSER 原始掩膜\n({len(regions)} 个区域)")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 6, 3)
    plt.title("3. 显著性热力图")
    plt.imshow(saliency_uint8, cmap='jet')
    plt.axis('off')

    plt.subplot(1, 6, 4)
    plt.title("4. 净化后掩膜")
    plt.imshow(clean_mask, cmap='gray')
    plt.axis('off')

    # 绘制边界框
    img_bbox = img.copy()
    for (x, y, wb, hb) in bboxes:
        cv2.rectangle(img_bbox, (x, y), (x + wb, y + hb), (0, 255, 0), 2)

    plt.subplot(1, 6, 5)
    plt.title("5. 边界框定位")
    plt.imshow(cv2.cvtColor(img_bbox, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 6, 6)
    plt.title("6. 叠加效果")
    overlay = cv2.addWeighted(img, 0.6, cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR), 0.4, 0)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    print(f"MSER 算法执行完毕，检测到 {len(regions)} 个稳定区域。")


if __name__ == "__main__":
    calculate_mser_saliency(local_path("test.jpg"))
