"""
教学示例：harris交叉点-T型交点

- 功能：演示 基元检测 中与“harris交叉点-T型交点”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""

import os

import cv2
import numpy as np


def nonmax_points(harris: np.ndarray, thresh: float) -> np.ndarray:
    mask = harris > thresh
    if not np.any(mask):
        return np.empty((0, 2), dtype=np.int32)

    dilated = cv2.dilate(harris, None)
    peaks = (harris == dilated) & mask
    ys, xs = np.where(peaks)
    return np.stack([xs, ys], axis=1).astype(np.int32)


def count_directions(gray: np.ndarray, edges: np.ndarray, x: int, y: int, r: int) -> int:
    h, w = gray.shape
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)

    patch = gray[y0:y1, x0:x1]
    edge_patch = edges[y0:y1, x0:x1]

    if patch.size == 0 or not np.any(edge_patch):
        return 0

    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = (cv2.phase(gx, gy, angleInDegrees=True)) % 180.0

    mags = mag[edge_patch > 0]
    angs = ang[edge_patch > 0]
    if mags.size == 0:
        return 0

    # Bin orientations into 9 bins (20 degrees each) over [0, 180)
    bins = np.floor(angs / 20.0).astype(np.int32)
    bins = np.clip(bins, 0, 8)

    hist = np.zeros(9, dtype=np.float32)
    for b, m in zip(bins, mags):
        hist[b] += m

    if hist.max() <= 0:
        return 0

    # Count dominant directions
    dominant = hist > (0.25 * hist.max())
    return int(np.count_nonzero(dominant))


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(here, "img.png")

    # Use imdecode to handle non-ASCII paths on Windows
    img_data = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_f = np.float32(gray)

    # Harris response
    harris = cv2.cornerHarris(gray_f, 2, 3, 0.04)
    harris = cv2.dilate(harris, None)
    thresh = 0.01 * harris.max()

    # Edge map for local direction analysis
    edges = cv2.Canny(gray, 80, 160)

    candidates = nonmax_points(harris, thresh)

    out = img.copy()
    t_points = []
    x_points = []

    for x, y in candidates:
        directions = count_directions(gray, edges, x, y, r=12)
        if directions == 3:
            t_points.append((x, y))
        elif directions >= 4:
            x_points.append((x, y))

    for x, y in t_points:
        cv2.circle(out, (x, y), 4, (0, 255, 255), 1)

    for x, y in x_points:
        cv2.circle(out, (x, y), 4, (0, 0, 255), 1)

    cv2.imshow("Harris T/Cross Points", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
