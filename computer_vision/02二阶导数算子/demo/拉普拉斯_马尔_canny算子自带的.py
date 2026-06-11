"""
教学示例：拉普拉斯 马尔 canny算子自带的

- 功能：演示 二阶导数与边缘检测 中与“拉普拉斯 马尔 canny算子自带的”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""
from pathlib import Path

import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt



BASE_DIR = Path(__file__).resolve().parent

def local_path(name: str) -> str:
    """返回脚本目录下的资源路径，避免依赖当前工作目录。"""
    return str(BASE_DIR / name)
# ==========================================
# 核心数据结构与算法辅助函数
# ==========================================

def pad_image(image, pad_size):
    """【算法】：为二维数组添加 padding，处理边界问题"""
    return np.pad(image, pad_size, mode='constant', constant_values=0)


def custom_convolve2d(image, kernel):
    """【算法】：手写二维卷积（滑动窗口操作）"""
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded_image = pad_image(image, (pad_h, pad_w))

    h, w = image.shape
    output = np.zeros((h, w), dtype=np.float32)

    # 遍历图像的每一个像素点 (O(N^2 * K^2) 时间复杂度)
    # 为保证运行时间在可接受范围内，这里使用 numpy 切片加速滑动窗口计算
    for i in range(h):
        for j in range(w):
            window = padded_image[i:i + k_h, j:j + k_w]
            output[i, j] = np.sum(window * kernel)

    return output


# ==========================================
# 1. Laplacian (拉普拉斯算子) - 手写实现
# ==========================================
def laplacian_algorithm(image):
    """手写拉普拉斯算子：使用3x3卷积核计算图像的二阶导数并取绝对值。"""
    # 定义 3x3 拉普拉斯卷积核
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)
    # 执行卷积运算
    lap = custom_convolve2d(image, kernel)
    return np.clip(np.abs(lap), 0, 255).astype(np.uint8)


# ==========================================
# 2. Marr-Hildreth (LoG) - 手写实现
# ==========================================
def marr_hildreth_algorithm(image):
    """手写 Marr-Hildreth (LoG) 算子：使用5x5高斯-拉普拉斯核卷积后取绝对值。"""
    # 定义 5x5 的高斯-拉普拉斯 (LoG) 核 (离散化逼近)
    log_kernel = np.array([
        [0, 0, -1, 0, 0],
        [0, -1, -2, -1, 0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1, 0],
        [0, 0, -1, 0, 0]
    ], dtype=np.float32)

    # 执行卷积
    log_out = custom_convolve2d(image, log_kernel)

    # 严格的 Marr 算法需要寻找“过零点”(Zero-crossing)
    # 为保持代码简洁，这里取绝对值阈值化来逼近零交叉附近的突变
    return np.clip(np.abs(log_out), 0, 255).astype(np.uint8)


# ==========================================
# 3. Canny 算子 - 从零手写完整算法管线
# ==========================================
def canny_algorithm(image, low_threshold=30, high_threshold=100):
    """手写 Canny 边缘检测：高斯平滑、Sobel梯度、非极大值抑制、滞后双阈值与BFS边缘连接。

    Args:
        image: 灰度输入图像。
        low_threshold: 弱边缘的低阈值，默认30。
        high_threshold: 强边缘的高阈值，默认100。
    """
    h, w = image.shape

    # Step 1: 高斯平滑 (简略化，使用 3x3 近似)
    gaussian_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    smoothed = custom_convolve2d(image, gaussian_kernel)

    # Step 2: 计算梯度幅值与方向 (Sobel)
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    Gx = custom_convolve2d(smoothed, Kx)
    Gy = custom_convolve2d(smoothed, Ky)

    # 梯度幅值 G = sqrt(Gx^2 + Gy^2)
    magnitude = np.hypot(Gx, Gy)
    magnitude = magnitude / magnitude.max() * 255.0  # 归一化到 0-255

    # 梯度方向 (角度转换为 0, 45, 90, 135 四个方向)
    angle = np.degrees(np.arctan2(Gy, Gx))
    angle[angle < 0] += 180

    # Step 3: 【算法核心】非极大值抑制 (NMS)
    nms = np.zeros((h, w), dtype=np.float32)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            q, r = 255.0, 255.0
            a = angle[i, j]

            # 根据梯度方向，找到前后两个相邻像素进行比较
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif (22.5 <= a < 67.5):
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif (67.5 <= a < 112.5):
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif (112.5 <= a < 157.5):
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            # 如果当前点是该方向上的极大值，则保留；否则抑制（置0）
            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                nms[i, j] = magnitude[i, j]
            else:
                nms[i, j] = 0

    # Step 4 & 5: 【数据结构核心】滞后双阈值化与 BFS 图搜索
    # 强边缘为 255，弱边缘为 50，非边缘为 0
    res = np.zeros((h, w), dtype=np.uint8)
    strong_i, strong_j = np.where(nms >= high_threshold)
    weak_i, weak_j = np.where((nms <= high_threshold) & (nms >= low_threshold))

    res[strong_i, strong_j] = 255
    res[weak_i, weak_j] = 50

    # 使用队列 (Queue) 进行广度优先搜索 (BFS)，连接边缘
    queue = deque()
    for i, j in zip(strong_i, strong_j):
        queue.append((i, j))

    # 八连通邻域偏移量
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while queue:
        cx, cy = queue.popleft()

        for dx, dy in neighbors:
            nx, ny = cx + dx, cy + dy
            # 检查边界
            if 0 <= nx < h and 0 <= ny < w:
                # 如果遇到弱边缘，将其提升为强边缘，并将其入队继续搜索
                if res[nx, ny] == 50:
                    res[nx, ny] = 255
                    queue.append((nx, ny))

    # 丢弃所有仍然是弱边缘的点（孤立点）
    res[res == 50] = 0
    return res


# ==========================================
# 主程序：执行与显示
# ==========================================
if __name__ == "__main__":
    print("正在从零开始计算二维卷积与图搜索算法，这可能需要大约 10-30 秒，请耐心等待...")

    img = cv2.imread(local_path("lena.png"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("错误：找不到 lena.png")
        exit()

    # 为防止 Python 手写循环耗时过长，我们缩小一下图片尺寸进行算法验证
    img_small = cv2.resize(img, (256, 256))

    # 执行手写的算法
    lap_out = laplacian_algorithm(img_small)
    log_out = marr_hildreth_algorithm(img_small)
    canny_out = canny_algorithm(img_small, low_threshold=20, high_threshold=80)

    # 显示结果
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(img_small, cmap='gray')
    plt.title('Original (256x256)')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(lap_out, cmap='gray')
    plt.title('Laplacian (Algo)')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(log_out, cmap='gray')
    plt.title('Marr-Hildreth (Algo)')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(canny_out, cmap='gray')
    plt.title('Canny (Algo + BFS)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    print("算法执行完毕！")