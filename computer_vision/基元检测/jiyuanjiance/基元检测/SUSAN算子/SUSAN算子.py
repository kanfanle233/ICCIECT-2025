# -*- coding:utf-8 -*-
"""
教学示例：SUSAN算子

- 功能：演示 基元检测 中与“SUSAN算子”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""
from pathlib import Path

import cv2
import numpy as np



BASE_DIR = Path(__file__).resolve().parent

def local_path(name: str) -> str:
    """返回脚本目录下的资源路径，避免依赖当前工作目录。"""
    return str(BASE_DIR / name)
def img_extraction(image, threshold_factor=10, geo_threshold=18):
    """
    SUSAN角点检测
    threshold_factor: 阈值因子（分母），越大检测越少
    geo_threshold: 几何阈值，越小检测越少
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 动态计算阈值
    threshold_value = int((gray.max() - gray.min()) / threshold_factor)
    print(f"灰度阈值: {threshold_value}, 几何阈值: {geo_threshold}")

    # 圆形模板偏移量
    offsetX = [
                -1, 0, 1,
            -2, -1, 0, 1, 2,
        -3, -2, -1, 0, 1, 2, 3,
        -3, -2, -1, 0, 1, 2, 3,
        -3, -2, -1, 0, 1, 2, 3,
            -2, -1, 0, 1, 2,
                -1, 0, 1
        ]
    offsetY = [
                -3, -3, -3,
            -2, -2, -2, -2, -2,
        -1, -1, -1, -1, -1, -1, -1,
             0, 0, 0, 0, 0, 0, 0,
             1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2,
                   3, 3, 3
        ]
    h, w = gray.shape
    result = np.zeros((h, w), dtype=np.uint8)

    for i in range(3, h - 3):
        for j in range(3, w - 3):
            same = 0
            nucleus = gray[i, j]
            for k in range(37):
                ni = i + offsetY[k]
                nj = j + offsetX[k]
                if abs(int(gray[ni, nj]) - int(nucleus)) < threshold_value:
                    same += 1
            if same < geo_threshold:
                result[i, j] = geo_threshold - same

    return result


def img_revise(response, min_distance=8, response_threshold=5):
    """
    非极大值抑制
    min_distance: 角点最小距离，越大角点越少
    response_threshold: 响应阈值，越大角点越少
    """
    h, w = response.shape

    # 收集所有候选点
    candidates = []
    for i in range(4, h - 4):
        for j in range(4, w - 4):
            if response[i, j] > response_threshold:
                candidates.append((j, i, response[i, j]))

    # 按响应值排序
    candidates.sort(key=lambda p: p[2], reverse=True)

    # 非极大值抑制
    corners = []
    for x, y, r in candidates:
        too_close = False
        for cx, cy in corners:
            if (x - cx) ** 2 + (y - cy) ** 2 < min_distance ** 2:
                too_close = True
                break
        if not too_close:
            corners.append((x, y))

    # 创建角点图像
    corner_img = np.zeros((h, w), dtype=np.uint8)
    for x, y in corners:
        corner_img[y, x] = 255

    return corner_img, corners


def mark_corners(image, corners):
    """在原图上标记角点"""
    result = image.copy()
    for x, y in corners:
        cv2.circle(result, (x, y), 3, (0, 0, 255), -1)
        cv2.circle(result, (x, y), 5, (0, 255, 255), 1)
    return result


if __name__ == '__main__':
    # 读取图像
    img = cv2.imread(local_path("img.png"))
    if img is None:
        print("使用测试图像")
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (250, 250), (255, 255, 255), 2)
        cv2.circle(img, (150, 150), 80, (255, 255, 255), 2)
        cv2.line(img, (50, 150), (250, 150), (255, 255, 255), 2)
        cv2.line(img, (150, 50), (150, 250), (255, 255, 255), 2)

    # ===== 参数调节区 =====
    # 调大以下参数可减少角点数量
    threshold_factor = 15  # 默认10，越大角点越少
    geo_threshold = 15  # 默认18，越小角点越少
    min_distance = 12  # 默认8，越大角点越少
    response_threshold = 8  # 默认5，越大角点越少
    # ====================

    print("=" * 40)
    print("当前参数设置:")
    print(f"threshold_factor: {threshold_factor}")
    print(f"geo_threshold: {geo_threshold}")
    print(f"min_distance: {min_distance}")
    print(f"response_threshold: {response_threshold}")
    print("=" * 40)

    # SUSAN角点检测
    response = img_extraction(img, threshold_factor, geo_threshold)

    # 非极大值抑制
    corners_img, corners = img_revise(response, min_distance, response_threshold)

    # 在原图上标记角点
    img_marked = mark_corners(img, corners)

    # 创建并排显示
    h, w = img.shape[:2]
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[0:h, 0:w] = img
    canvas[0:h, w:w * 2] = img_marked

    # 添加文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, 'Original', (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(canvas, f'Corners: {len(corners)}', (w + 10, 30), font, 1, (0, 255, 0), 2)

    # 显示
    cv2.imshow('SUSAN Corner Detection', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"检测到角点数量：{len(corners)}")

    # 灰度阈值和几何阈值的作用（通俗版）
    # 1.
    # 灰度阈值（T）—— 判断“像不像”
    # python
    # # 代码中的体现
    # if abs(邻居灰度 - 中心灰度) < 灰度阈值:
    #     认为是“相似的”
    #     作用：决定什么样的邻居算“和自己像”
    #
    #     灰度阈值
    #     效果
    #     比喻
    #     调大
    #     更多邻居被算作“相似”    更宽容，看谁都像亲戚
    #     调小
    #     更少邻居被算作“相似”    更严格，只有非常像才算
    #     例子：
    #
    #     中心像素 = 100
    #
    #     灰度阈值 = 20 → 80
    #     ~120
    #     的都算相似
    #
    #     灰度阈值 = 10 → 90
    #     ~110
    #     的才算相似
    #
    #     2.
    #     几何阈值（G）—— 判断“是不是角点”
    #     python
    #     # 代码中的体现
    #     if 相似邻居数量 < 几何阈值:
    #         认为是角点
    #         response = 几何阈值 - 相似邻居数量
    #     作用：决定相似邻居多少个才算角点
    #
    #     几何阈值
    #     效果
    #     比喻
    #     调大
    #     更容易被判定为角点
    #     疑心重，稍有不同就算角点
    #     调小
    #     更难被判定为角点
    #     要求高，必须非常不同才算
    #     3.
    #     两者配合的效果
    #     情况
    #     灰度阈值
    #     几何阈值
    #     结果
    #     严格
    #     小
    #     小
    #     角点很少（只留最明显的）
    #     宽松
    #     大
    #     大
    #     角点很多（包括边缘和噪声）
    #     平衡
    #     适中
    #     适中
    #     只留真正的角点
    #     4.
    #     直观理解
    #     USAN面积 = 相似邻居的数量
    #
    #     平坦区域：几乎所有邻居都相似 → USAN≈37 → 不是角点
    #
    #     边缘上：大约一半相似 → USAN≈18 - 20 → 弱响应
    #
    #     角点上：很少相似 → USAN≈10 - 15 → 强响应
    #
    #     几何阈值G就是用来划这条线：
    #
    #     如果USAN < G → 判定为角点
    #
    #     G越小，要求USAN越小，即要求中心点必须非常“与众不同”
    #
    #     5.
    #     当前参数的意义
    #     text
    #     灰度阈值 = 16  → 灰度差小于16的算相似
    #     几何阈值 = 15  → 相似邻居少于15个才算角点
    #     这意味着：
    #
    #     中心像素要找那些和自己差别小于16的邻居
    #
    #     如果这样的邻居少于15个，就认为这是个角点
    #
    #     响应强度 = 15 - 相似邻居数量

