#!/usr/bin/env python3
"""
教学示例：skeleton compare

- 功能：演示 目标区域表达与骨架提取 中与“skeleton compare”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""

# Compare multiple skeletonization methods on img.png

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(f"OpenCV (cv2) is required: {exc}")

try:
    from skimage.morphology import medial_axis, skeletonize, thin

    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False


def to_binary(gray: np.ndarray) -> np.ndarray:
    """Otsu binarization with automatic foreground polarity."""
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # If foreground is likely dark, invert
    if np.mean(bw == 255) > 0.5:
        bw = cv2.bitwise_not(bw)
    return (bw > 0).astype(np.uint8)


def zhang_suen_thinning(img: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning on binary image (0/1)."""
    out = img.copy().astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (1, 2):
            p = np.pad(out, ((1, 1), (1, 1)), mode="constant")
            p2 = p[:-2, 1:-1]  # N
            p3 = p[:-2, 2:]    # NE
            p4 = p[1:-1, 2:]   # E
            p5 = p[2:, 2:]     # SE
            p6 = p[2:, 1:-1]   # S
            p7 = p[2:, :-2]    # SW
            p8 = p[1:-1, :-2]  # W
            p9 = p[:-2, :-2]   # NW

            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                (p2 == 0) & (p3 == 1) +
                (p3 == 0) & (p4 == 1) +
                (p4 == 0) & (p5 == 1) +
                (p5 == 0) & (p6 == 1) +
                (p6 == 0) & (p7 == 1) +
                (p7 == 0) & (p8 == 1) +
                (p8 == 0) & (p9 == 1) +
                (p9 == 0) & (p2 == 1)
            )

            if step == 1:
                c1 = (p2 * p4 * p6) == 0
                c2 = (p4 * p6 * p8) == 0
            else:
                c1 = (p2 * p4 * p8) == 0
                c2 = (p2 * p6 * p8) == 0

            cond = (
                (out == 1)
                & (neighbors >= 2)
                & (neighbors <= 6)
                & (transitions == 1)
                & c1
                & c2
            )
            if np.any(cond):
                out[cond] = 0
                changed = True
    return out


def skimage_skeleton(img: np.ndarray) -> np.ndarray:
    """使用 scikit-image 的 skeletonize 提取骨架，不可用时回退到 Zhang-Suen。"""
    if not HAS_SKIMAGE:
        return zhang_suen_thinning(img)
    return skeletonize(img > 0).astype(np.uint8)


def skimage_thin(img: np.ndarray) -> np.ndarray:
    """使用 scikit-image 的 thin 提取骨架，不可用时回退到 Zhang-Suen。"""
    if not HAS_SKIMAGE:
        return zhang_suen_thinning(img)
    return thin(img > 0).astype(np.uint8)


def medial_axis_skeleton(img: np.ndarray) -> np.ndarray:
    """使用 scikit-image 的 medial_axis 提取中轴骨架，不可用时回退到 Zhang-Suen。"""
    if not HAS_SKIMAGE:
        return zhang_suen_thinning(img)
    skel, _ = medial_axis(img > 0, return_distance=True)
    return skel.astype(np.uint8)


def opencv_distance_transform_skeleton(img: np.ndarray) -> np.ndarray:
    """Use OpenCV distance transform and ridge extraction."""
    src = (img > 0).astype(np.uint8) * 255
    dt = cv2.distanceTransform(src, cv2.DIST_L2, 5)
    dt_dil = cv2.dilate(dt, np.ones((3, 3), np.uint8))
    ridge = (dt > 0) & (dt >= dt_dil - 1e-6)
    ridge = ridge.astype(np.uint8)
    return zhang_suen_thinning(ridge)


def pyramid_skeleton(img: np.ndarray, levels: int = 3) -> np.ndarray:
    """Multi-scale pyramid skeletonization, merged and thinned."""
    pyr = [img.astype(np.uint8)]
    for _ in range(levels):
        down = cv2.pyrDown(pyr[-1].astype(np.uint8) * 255)
        pyr.append((down > 0).astype(np.uint8))

    merged = np.zeros_like(img, dtype=np.uint8)
    for level, bi in enumerate(pyr):
        sk = skimage_skeleton(bi)
        up = sk.astype(np.uint8)
        for _ in range(level):
            up = cv2.pyrUp(up.astype(np.uint8))
        up = cv2.resize(up, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        merged = np.logical_or(merged, up).astype(np.uint8)

    return zhang_suen_thinning(merged)


def quadtree_skeleton(img: np.ndarray, min_size: int = 8, occ_thresh: float = 0.15) -> np.ndarray:
    """Simple quadtree-based centerline approximation."""
    h, w = img.shape
    leaves = []

    def split(x: int, y: int, size: int):
        patch = img[y:y + size, x:x + size]
        if patch.size == 0:
            return
        occ = np.mean(patch > 0)
        if size <= min_size or (occ < occ_thresh) or (occ > (1 - occ_thresh)):
            if occ > occ_thresh:
                leaves.append((x, y, size))
            return
        half = size // 2
        if half < 1:
            return
        split(x, y, half)
        split(x + half, y, half)
        split(x, y + half, half)
        split(x + half, y + half, half)

    size = 2 ** int(math.floor(math.log2(min(h, w))))
    split(0, 0, size)

    canvas = np.zeros((h, w), dtype=np.uint8)

    def center(box):
        """计算四叉树叶节点 (x, y, size) 的中心坐标。"""
        x, y, s = box
        return x + s // 2, y + s // 2

    for i, a in enumerate(leaves):
        ax, ay, asz = a
        acx, acy = center(a)
        for b in leaves[i + 1:]:
            bx, by, bsz = b
            bcx, bcy = center(b)
            # Adjacent if edges touch with overlap
            touch_x = (ax + asz == bx) or (bx + bsz == ax)
            overlap_y = not (ay + asz <= by or by + bsz <= ay)
            touch_y = (ay + asz == by) or (by + bsz == ay)
            overlap_x = not (ax + asz <= bx or bx + bsz <= ax)
            if (touch_x and overlap_y) or (touch_y and overlap_x):
                cv2.line(canvas, (acx, acy), (bcx, bcy), 1, 1)

        cv2.circle(canvas, (acx, acy), 1, 1, -1)

    return zhang_suen_thinning(canvas)


def make_tile(binary: np.ndarray, label: str, tile_size: tuple[int, int], label_h: int = 36) -> np.ndarray:
    """将二值图像缩放到指定尺寸并添加顶部文字标签，生成展示用瓦片。"""
    h, w = tile_size
    base = np.full((h + label_h, w, 3), 255, dtype=np.uint8)
    disp = 255 - (binary.astype(np.uint8) * 255)
    disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_NEAREST)
    base[label_h:, :, :] = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        base,
        label,
        (10, label_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return base


def build_grid(tiles: list[np.ndarray], cols: int = 3, pad: int = 10) -> np.ndarray:
    """将多个瓦片图像按指定列数和间距拼接成网格图。"""
    rows = math.ceil(len(tiles) / cols)
    tile_h, tile_w = tiles[0].shape[:2]
    grid = np.full(
        (rows * tile_h + (rows + 1) * pad, cols * tile_w + (cols + 1) * pad, 3),
        245,
        dtype=np.uint8,
    )
    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        y0 = pad + r * (tile_h + pad)
        x0 = pad + c * (tile_w + pad)
        grid[y0:y0 + tile_h, x0:x0 + tile_w] = tile
    return grid


def main() -> None:
    """执行所有骨架提取方法并生成对比网格图，保存到 outputs 目录。"""
    here = Path(__file__).resolve().parent
    img_path = here / "img.png"
    if not img_path.exists():
        raise SystemExit(f"Missing image: {img_path}")

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise SystemExit(f"Failed to read: {img_path}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    binary = to_binary(gray)

    methods = [
        ("binary", binary),
        ("zhang-suen", zhang_suen_thinning(binary)),
        ("dist-medial-axis", medial_axis_skeleton(binary)),
        ("quadtree", quadtree_skeleton(binary)),
        ("pyramid", pyramid_skeleton(binary)),
        ("opencv-dist", opencv_distance_transform_skeleton(binary)),
        ("skimage-skeleton", skimage_skeleton(binary)),
        ("skimage-thin", skimage_thin(binary)),
    ]

    out_dir = here / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, img in methods:
        cv2.imwrite(str(out_dir / f"{name}.png"), 255 - (img.astype(np.uint8) * 255))

    tile_w = min(480, binary.shape[1])
    tile_h = int(tile_w * binary.shape[0] / binary.shape[1])
    tiles = [make_tile(img, name, (tile_h, tile_w)) for name, img in methods]
    grid = build_grid(tiles, cols=3, pad=12)
    cv2.imwrite(str(out_dir / "skeleton_compare_grid.png"), grid)

    print("Done.")
    print(f"Outputs: {out_dir}")
    if not HAS_SKIMAGE:
        print("Note: scikit-image not found; some methods fell back to Zhang-Suen.")


if __name__ == "__main__":
    main()
