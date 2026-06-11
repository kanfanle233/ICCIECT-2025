"""
教学示例：saliency rc

- 功能：演示 显著性检测 中与“saliency rc”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.color import rgb2lab


BASE_DIR = Path(__file__).resolve().parent

def local_path(name: str) -> str:
    """返回脚本目录下的资源路径，避免依赖当前工作目录。"""
    return str(BASE_DIR / name)
# 1. 设置字体为 SimHei (黑体)，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
# 2. 解决保存图像时负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False


def calculate_rc_saliency(image_path=local_path("test.jpg"), n_segments=200):
    """
    RC (Region Contrast): 基于区域对比度的显著性检测
    原理: 将图像分割为超像素，计算每个区域与其他所有区域的颜色距离和空间距离加权和
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取 {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # 1. SLIC 超像素分割
    # compactness 越高，超像素越规则
    segments = slic(img_rgb, n_segments=n_segments, compactness=10, start_label=0)

    # 2. 计算每个区域的特征 (平均颜色 Lab, 像素数, 中心坐标)
    # 转换到 Lab 色彩空间以更好地感知颜色差异
    img_lab = rgb2lab(img_rgb)

    region_props = {}
    unique_ids = np.unique(segments)

    for id in unique_ids:
        mask = (segments == id)
        coords = np.column_stack(np.where(mask))

        # 平均颜色
        mean_color = np.mean(img_lab[mask], axis=0)
        # 区域大小
        area = np.sum(mask)
        # 区域中心
        center_y, center_x = np.mean(coords, axis=0)

        region_props[id] = {
            'color': mean_color,
            'area': area,
            'center': (center_x, center_y),
            'mask': mask
        }

    total_area = h * w
    n_regions = len(unique_ids)

    # 3. 计算每个区域的显著性
    # S(R_i) = sum( D_color(R_i, R_j) * D_space(R_i, R_j) * F(R_j) )
    region_saliency = np.zeros(n_regions, dtype=np.float32)

    # 预计算空间距离权重参数 (sigma_s)
    sigma_s = np.sqrt(w ** 2 + h ** 2) * 0.5

    ids_list = list(unique_ids)

    for i, id_i in enumerate(ids_list):
        prop_i = region_props[id_i]
        s_val = 0.0

        for j, id_j in enumerate(ids_list):
            if i == j: continue

            prop_j = region_props[id_j]

            # 颜色距离 (欧氏距离 in Lab)
            d_color = np.linalg.norm(prop_i['color'] - prop_j['color'])

            # 空间距离 (高斯权重)
            dist_center = np.linalg.norm(np.array(prop_i['center']) - np.array(prop_j['center']))
            d_space = np.exp(- (dist_center ** 2) / (2 * sigma_s ** 2))

            # 频率权重
            freq = prop_j['area'] / total_area

            s_val += d_color * d_space * freq

        region_saliency[i] = s_val

    # 4. 生成显著性图
    saliency_map = np.zeros((h, w), dtype=np.float32)
    for i, id in enumerate(ids_list):
        mask = region_props[id]['mask']
        saliency_map[mask] = region_saliency[i]

    # 归一化
    saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)
    saliency_uint8 = saliency_map.astype(np.uint8)

    # 可视化
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 5, 1)
    plt.title("1. 原始图像")
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.title(f"2. 超像素分割\n(n={n_segments})")
    plt.imshow(segments, cmap='nipy_spectral')
    plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.title("3. RC 显著性图")
    plt.imshow(saliency_uint8, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.title("4. 平滑后显著性图")
    # 由于超像素是块状的，通常用原图引导滤波或简单高斯模糊
    blurred = cv2.GaussianBlur(saliency_uint8, (5, 5), 0)
    plt.imshow(blurred, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.title("5. 二值化结果 (Otsu)")
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.imshow(binary, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    print("RC 算法执行完毕。")


if __name__ == "__main__":
    calculate_rc_saliency(local_path("test.jpg"))
