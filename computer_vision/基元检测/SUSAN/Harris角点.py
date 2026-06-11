"""
教学示例：Harris角点

- 功能：演示 基元检测 中与“Harris角点”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""

import os

import cv2
import numpy as np


def main() -> None:
    """读取图像并执行Harris角点检测，在图像上标记角点并显示结果。"""
    here = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(here, "img.png")

    # Use imdecode to handle non-ASCII paths on Windows
    img_data = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_f = np.float32(gray)

    # Harris corner detection
    block_size = 2
    ksize = 3
    k = 0.04
    harris = cv2.cornerHarris(gray_f, block_size, ksize, k)
    harris = cv2.dilate(harris, None)

    # Threshold and draw corners
    thresh = 0.01 * harris.max()
    out = img.copy()
    out[harris > thresh] = (0, 0, 255)

    cv2.imshow("Harris Corners", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
