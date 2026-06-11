"""
教学示例：susan edge

- 功能：演示 基元检测 中与“susan edge”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time


BASE_DIR = Path(__file__).resolve().parent

def local_path(name: str) -> str:
    """返回脚本目录下的资源路径，避免依赖当前工作目录。"""
    return str(BASE_DIR / name)
# 【关键步骤 1】强制设置后端为 TkAgg (通用性最好) 或 Qt5Agg
# 这行代码必须在 import matplotlib.pyplot 之前或之后立即运行
# 如果在 PyCharm 中仍然不弹出，请尝试取消注释下一行并注释掉 TkAgg
try:
    import tkinter

    plt.switch_backend('TkAgg')  # 使用 Tkinter 后端，通常能弹出独立窗口
except ImportError:
    # 如果没有 tkinter，尝试 Qt
    try:
        plt.switch_backend('Qt5Agg')
    except:
        pass

# 【关键步骤 2】配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False


def susan_edge_detection(image, threshold=50, geometric_threshold=0.75):
    """
    手动实现 SUSAN 边缘检测算法
    """
    h, w = image.shape
    output = np.zeros((h, w), dtype=np.uint8)

    radius = 3
    # 生成圆形掩模
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = (x * x + y * y) <= (radius * radius)

    n_max = np.sum(mask)
    g = n_max * geometric_threshold

    # 边界填充
    padded_img = cv2.copyMakeBorder(image, radius, radius, radius, radius, cv2.BORDER_REPLICATE)

    print(f"正在计算 SUSAN (图像尺寸: {w}x{h})...")

    # 优化：使用局部变量加速循环
    for i in range(h):
        for j in range(w):
            window = padded_img[i:i + 2 * radius + 1, j:j + 2 * radius + 1]
            center_val = window[radius, radius]

            # 计算差异
            diff = np.abs(window.astype(np.int16) - center_val)
            masked_diff = diff[mask]

            usan = np.sum(masked_diff < threshold)

            if usan < g:
                output[i, j] = 255
            else:
                output[i, j] = 0

    return output


def main():
    """读取图像并执行SUSAN边缘检测，与Canny结果对比显示。"""
    image_path = local_path("test_image.jpg")
    if not os.path.exists(image_path):
        print(f"错误：找不到文件 {image_path}")
        print("请在当前目录下放入一张名为 test.jpg 的图片。")
        return

    # 读取图像
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("无法读取图片，请检查文件格式。")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 执行算法
    start_time = time.time()
    susan_edges = susan_edge_detection(gray, threshold=40, geometric_threshold=0.6)
    end_time = time.time()

    canny_edges = cv2.Canny(gray, 50, 150)

    # --- 绘图部分 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.canvas.manager.set_window_title('SUSAN 边缘检测独立演示窗口')  # 设置窗口标题
    fig.suptitle('SUSAN vs Canny 边缘检测对比', fontsize=16, fontweight='bold')

    # 1. 原图
    axes[0].imshow(img_rgb)
    axes[0].set_title('原始图像')
    axes[0].axis('off')

    # 2. SUSAN
    axes[1].imshow(susan_edges, cmap='gray')
    axes[1].set_title(f'SUSAN 结果 (耗时: {end_time - start_time:.2f}s)')
    axes[1].axis('off')

    # 3. Canny
    axes[2].imshow(canny_edges, cmap='gray')
    axes[2].set_title('Canny 结果 (对比)')
    axes[2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    print(">>> 正在弹出独立窗口，请查看任务栏...")
    print(">>> 关闭窗口后程序将结束。")

    # 【关键步骤 3】show() 会阻塞程序并弹出窗口
    plt.show()

    print("窗口已关闭，程序结束。")


if __name__ == "__main__":
    main()