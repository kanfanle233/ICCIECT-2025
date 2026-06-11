"""
教学示例：position histogram track

- 功能：演示 基元检测 中与“position histogram track”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""

import cv2
import numpy as np
import math


def create_gaussian_weight_map(shape, sigma=0.5):
    """创建一个以图像中心为峰值的二维高斯权重图。"""
    rows, cols = shape
    center_x, center_y = cols / 2, rows / 2

    x = np.linspace(0, cols - 1, cols)
    y = np.linspace(0, rows - 1, rows)
    X, Y = np.meshgrid(x, y)

    dist_sq = (X - center_x) ** 2 + (Y - center_y) ** 2
    sigma_pixels = min(rows, cols) * sigma
    weight_map = np.exp(-dist_sq / (2 * sigma_pixels ** 2))

    return weight_map.astype(np.float32)  # 确保权重图也是 float32


def calculate_spatial_histogram(image, mask, bins=16):
    """计算带有高斯位置权重的HSV色调直方图，返回float32列向量。"""
    h, w = image.shape[:2]

    # 1. 创建位置权重图
    weight_map = create_gaussian_weight_map((h, w), sigma=0.4)

    if mask is not None:
        mask_norm = mask.astype(np.float32) / 255.0
        final_weights = weight_map * mask_norm
    else:
        final_weights = weight_map

    # 2. 转换为 HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0].astype(np.float32)

    bin_width = 180 // bins

    # 初始化直方图 (使用 float32)
    hist = np.zeros(bins, dtype=np.float32)

    # 优化：使用向量化操作代替双重循环，速度更快且不易出错
    # 将权重和颜色通道展平
    weights_flat = final_weights.ravel()
    h_flat = h_channel.ravel()

    # 计算每个像素对应的 bin 索引
    bin_indices = (h_flat / bin_width).astype(np.int32)
    bin_indices = np.clip(bin_indices, 0, bins - 1)

    # 累加权重到对应的 bin
    # np.add.at 可以处理重复索引的累加
    np.add.at(hist, bin_indices, weights_flat)

    # 归一化
    total = np.sum(hist)
    if total > 0:
        hist /= total

    # 【关键修复】：OpenCV compareHist 需要 float32 类型的列向量 (shape: (N, 1)) 或行向量
    return hist.reshape(-1, 1).astype(np.float32)


def compare_histograms(hist1, hist2):
    """使用巴氏距离比较两个直方图的相似度，返回0~1的相似度值。"""
    # 再次确保类型和形状正确（防御性编程）
    if hist1.dtype != np.float32:
        hist1 = hist1.astype(np.float32)
    if hist2.dtype != np.float32:
        hist2 = hist2.astype(np.float32)

    # 如果形状是 (N,) 变成 (N, 1)
    if len(hist1.shape) == 1:
        hist1 = hist1.reshape(-1, 1)
    if len(hist2.shape) == 1:
        hist2 = hist2.reshape(-1, 1)

    # 使用巴氏距离 (Bhattacharyya Distance)
    # 返回值范围 0 (完全匹配) 到 1 (完全不匹配)
    try:
        distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        similarity = 1 - distance
        return max(0.0, similarity)  # 防止出现微小的负数
    except Exception as e:
        print(f"比较直方图时出错: {e}")
        print(f"Hist1 Shape: {hist1.shape}, Type: {hist1.dtype}")
        print(f"Hist2 Shape: {hist2.shape}, Type: {hist2.dtype}")
        return 0.0


# ================= 主程序 =================

def main():
    """创建模拟场景，通过位置直方图匹配在全图扫描中定位目标。"""
    # 1. 创建一个模拟视频帧 (黑色背景)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 2. 绘制目标 (中心红色方块)
    target_center_x = 320
    target_y = 240
    target_size = 50

    cv2.rectangle(frame,
                  (target_center_x - target_size, target_y - target_size),
                  (target_center_x + target_size, target_y + target_size),
                  (0, 0, 255), -1)

    # 3. 绘制干扰物 (边缘红色圆)
    cv2.circle(frame, (50, 50), 40, (0, 0, 255), -1)
    cv2.circle(frame, (590, 430), 40, (0, 0, 255), -1)

    cv2.putText(frame, "Target (Center)", (target_center_x - 60, target_y - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Distractors (Edges)", (200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 4. 定义目标的掩膜
    mask = np.zeros_like(frame[:, :, 0])
    cv2.rectangle(mask,
                  (target_center_x - target_size, target_y - target_size),
                  (target_center_x + target_size, target_y + target_size),
                  255, -1)

    print("正在计算参考模型...")
    # 5. 计算参考直方图
    ref_hist = calculate_spatial_histogram(frame, mask, bins=16)
    print(f"参考直方图形状: {ref_hist.shape}, 类型: {ref_hist.dtype}")

    # 6. 扫描搜索
    print("正在扫描全图...")

    best_score = -1
    best_loc = None

    win_h, win_w = 100, 100
    step = 20

    # 预分配一些变量以加速
    h_limit = frame.shape[0] - win_h
    w_limit = frame.shape[1] - win_w

    for y in range(0, h_limit, step):
        for x in range(0, w_limit, step):
            roi = frame[y:y + win_h, x:x + win_w]

            # 计算当前窗口的直方图 (假设目标在窗口中心，所以 mask 为 None，依靠内部高斯权重)
            current_hist = calculate_spatial_histogram(roi, None, bins=16)

            score = compare_histograms(ref_hist, current_hist)

            if score > best_score:
                best_score = score
                best_loc = (x, y)

    # 7. 可视化结果
    result_frame = frame.copy()
    if best_loc:
        bx, by = best_loc
        cv2.rectangle(result_frame, (bx, by), (bx + win_w, by + win_h), (0, 255, 0), 2)
        label = f"Score: {best_score:.2f}"
        cv2.putText(result_frame, label, (bx, by - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        print(f"\n成功找到目标！")
        print(f"最佳位置: ({bx}, {by})")
        print(f"相似度得分: {best_score:.4f} (越接近 1.0 越好)")
        print("绿色框应准确框住中间的红色方块，忽略边缘的红色圆圈。")
    else:
        print("未找到匹配项。")

    cv2.imshow("Position Histogram Demo", result_frame)
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()