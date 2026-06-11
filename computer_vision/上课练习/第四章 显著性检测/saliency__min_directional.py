"""
教学示例：saliency  min directional

- 功能：演示 显著性检测 中与“saliency  min directional”相关的核心流程。
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

def calculate_min_directional_contrast(image_path, kernel_size=5):
    """
    基于最小方向对比度 (Minimum Directional Contrast, MDC) 的显著性检测
    原理: 计算像素在多个方向上的局部对比度，取最小值以抑制单向纹理背景。
    """
    # 1. 读取图像并转为灰度 (方向对比度通常在亮度通道计算最有效)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError("无法读取图像")

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = img_gray.shape

    # 2. 定义方向算子 (4个方向: 0°, 45°, 90°, 135°)
    # 这里使用简单的中心差分思想：计算中心像素与周围像素在特定方向上的最大差值
    # 为了简化，我们构建4个方向的差分核，或者直接滑动窗口计算

    # 初始化存储4个方向对比度的列表
    directional_contrasts = []

    # 定义半核大小
    half_k = kernel_size // 2

    # 为了提高速度，使用卷积方式近似计算方向对比度
    # 构造4个方向的差分滤波器 (中心为0, 两端为1和-1的变体，或者简单的梯度算子)
    # 这里采用一种更直观的方法：计算中心区域与四个方向邻域的平均值之差的绝对值

    # 方法：对于每个方向，计算 "中心像素值" 与 "该方向上邻域像素平均值" 的差
    # 实际上，更高效的是使用预定义的卷积核来提取特定方向的边缘强度，然后取最小值是不对的。
    # 修正逻辑：MDC 通常指 "Local Contrast in Direction d" = |I(center) - I(neighbor_d)|
    # 我们计算4个方向的局部反差，然后取 min。

    # 让我们构建4个方向的掩膜，用于计算中心与特定方向邻域的差
    # 方向定义:
    # 0: 水平 (左/右)
    # 1: 垂直 (上/下)
    # 2: 对角线 (左上/右下)
    # 3: 反对角线 (右上/左下)

    contrasts = np.zeros((h, w, 4), dtype=np.float32)

    # 使用卷积来加速计算 "中心与邻域的差"
    # 构造核：中心为1，邻域为 -1/N (N为邻域像素数)
    # 但为了简单且符合“对比度”定义，我们直接计算 |Center - Mean(Direction)|

    # 为了代码简洁和效率，这里使用形态学或卷积近似
    # 定义4个方向的线性结构元素，用于获取邻域均值
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1)),  # 水平
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size)),  # 垂直
        None,  # 45度 (需自定义)
        None  # 135度 (需自定义)
    ]

    # 自定义对角线核
    diag_k = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    for i in range(kernel_size):
        diag_k[i, i] = 1
    kernels[2] = diag_k

    anti_diag_k = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    for i in range(kernel_size):
        anti_diag_k[i, kernel_size - 1 - i] = 1
    kernels[3] = anti_diag_k

    # 计算每个方向的邻域均值
    means = []
    for k in kernels:
        # 注意：filter2D 默认处理边界，这里简单处理
        mean_img = cv2.filter2D(img_gray, -1, k / np.sum(k))
        means.append(mean_img)

    # 计算中心像素 (可以用原图近似，或者也用一个小核平滑一下中心)
    # 严格来说，中心就是 img_gray 本身 (或者去除噪声后的版本)
    # 对比度 = | Center - Mean_Direction |
    # 为了防止中心本身就是噪点，我们可以用 3x3 均值代表中心
    center_mean = cv2.blur(img_gray, (3, 3))

    for i in range(4):
        diff = cv2.absdiff(center_mean, means[i])
        contrasts[:, :, i] = diff

    # 3. 取最小方向对比度 (Minimum Directional Contrast)
    # 轴2是方向维度
    mdc_map = np.min(contrasts, axis=2)

    # 4. 归一化
    mdc_map = cv2.normalize(mdc_map, None, 0, 255, cv2.NORM_MINMAX)
    mdc_map = mdc_map.astype(np.uint8)

    # 5. 后处理
    # MDC 图可能比较细碎，因为它是基于局部差分的。
    # 通常需要结合全局信息或进行较强的平滑/形态学操作来形成完整的显著区域
    # 这里应用高斯模糊和形态学闭运算来连接区域
    mdc_blur = cv2.GaussianBlur(mdc_map, (7, 7), 0)

    # 可选：自适应阈值或 Otsu
    _, binary_mask = cv2.threshold(mdc_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 为了效果更好，可以结合之前的 FT (全局分布) 结果
    # 纯 MDC 有时只能提取出物体的“角点”或“各向异性”部分，
    # 真正的显著性检测常将 MDC 作为权重或特征之一。
    # 但作为独立演示，我们展示纯 MDC 的效果。

    return img_bgr, mdc_map, mdc_blur, binary_mask, contrasts


# --- 主程序 ---
if __name__ == "__main__":
    image_file = local_path("22.jpg")

    import os

    if not os.path.exists(image_file):
        print(f"未找到 {image_file}，正在生成纹理背景测试图像...")
        h, w = 400, 400
        test_img = np.ones((h, w, 3), dtype=np.uint8) * 200

        # 制造强烈的单向纹理背景 (横线)
        for i in range(0, h, 4):
            # 画深灰色的横线
            cv2.line(test_img, (0, i), (w, i), (100, 100, 100), 1)

        # 添加一些纵向的弱纹理，增加复杂度
        for i in range(0, w, 20):
            cv2.line(test_img, (i, 0), (i, h), (180, 180, 180), 1)

        # 前景：一个实心圆 (各向同性，无方向性纹理)
        cv2.circle(test_img, (200, 200), 50, (0, 0, 255), -1)  # 红色圆

        # 前景：一个方块
        cv2.rectangle(test_img, (80, 80), (140, 140), (0, 255, 0), -1)  # 绿色方块

        cv2.imwrite(image_file, test_img)
        print("测试图像已生成 (横线背景 + 圆形/方形前景)。")

    try:
        original, mdc_raw, mdc_blur, mask, all_contrasts = calculate_min_directional_contrast(image_file, kernel_size=9)

        plt.figure(figsize=(20, 4))

        plt.subplot(1, 7, 1)
        plt.title("1. 原始图像\n(横线背景+物体)")
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        # 展示4个方向的对比度图，帮助理解
        directions = ["0° (水平)", "90° (垂直)", "45° (对角)", "135° (反对角)"]
        for i in range(4):
            plt.subplot(1, 7, i + 2)
            plt.title(f"{i + 2}. 方向对比度: {directions[i]}")
            plt.imshow(all_contrasts[:, :, i], cmap='gray')
            plt.axis('off')

        plt.subplot(1, 7, 6)
        plt.title("6. 最小方向对比度 (MDC)\n取4个方向的最小值")
        plt.imshow(mdc_raw, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 7, 7)
        plt.title("7. 最终二值化结果\n(平滑后)")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        print("分析提示：")
        print("- 在'水平方向对比度'图中，横线背景应该是黑的（因为水平方向无变化）。")
        print("- 在'垂直方向对比度'图中，横线背景应该是白的（因为垂直方向变化大）。")
        print("- 在'MDC (最小值)'图中，横线背景应该消失（因为取最小值，0和大的数取0）。")
        print("- 圆形和方形在所有方向都有对比度，所以在 MDC 图中保留了下来。")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback

        traceback.print_exc()