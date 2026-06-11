"""
教学示例：202314109方昕哲

- 功能：演示 目标区域表达与骨架提取 中与“202314109方昕哲”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math

BASE_DIR = Path(__file__).resolve().parent

def local_path(name: str) -> str:
    """返回脚本目录下的资源路径，避免依赖当前工作目录。"""
    return str(BASE_DIR / name)
try:
    from skimage.morphology import medial_axis, skeletonize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# 设置字体以支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

def to_binary(gray):
    """自适应大津法二值化，并自动根据背景极性翻转"""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 如果大部分像素是白色的，可能背景被识别为了白色，因此反转
    if np.mean(bw == 255) > 0.5:
        bw = cv2.bitwise_not(bw)
    return (bw > 0).astype(np.uint8)

def zhang_suen_thinning(img):
    """Zhang-Suen 细化算法纯Python实现"""
    out = img.copy().astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (1, 2):
            p = np.pad(out, ((1, 1), (1, 1)), mode="constant")
            p2 = p[:-2, 1:-1]
            p3 = p[:-2, 2:]
            p4 = p[1:-1, 2:]
            p5 = p[2:, 2:]
            p6 = p[2:, 1:-1]
            p7 = p[2:, :-2]
            p8 = p[1:-1, :-2]
            p9 = p[:-2, :-2]

            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = ((p2 == 0) & (p3 == 1)).astype(int) + \
                          ((p3 == 0) & (p4 == 1)).astype(int) + \
                          ((p4 == 0) & (p5 == 1)).astype(int) + \
                          ((p5 == 0) & (p6 == 1)).astype(int) + \
                          ((p6 == 0) & (p7 == 1)).astype(int) + \
                          ((p7 == 0) & (p8 == 1)).astype(int) + \
                          ((p8 == 0) & (p9 == 1)).astype(int) + \
                          ((p9 == 0) & (p2 == 1)).astype(int)

            if step == 1:
                c1 = (p2 * p4 * p6) == 0
                c2 = (p4 * p6 * p8) == 0
            else:
                c1 = (p2 * p4 * p8) == 0
                c2 = (p2 * p6 * p8) == 0

            cond = (out == 1) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & c1 & c2
            if np.any(cond):
                out[cond] = 0
                changed = True
    return out

def opencv_dt_skeleton(img):
    """基于OpenCV的距离变换脊线提取骨架"""
    dt = cv2.distanceTransform(img * 255, cv2.DIST_L2, 5)
    dt_dil = cv2.dilate(dt, np.ones((3, 3), np.uint8))
    ridge = (dt > 0) & (dt >= dt_dil - 1e-6)
    return zhang_suen_thinning(ridge.astype(np.uint8))

def pyramid_skeleton(img, levels=3):
    """基于金字塔的多尺度骨架提取"""
    pyr = [img.astype(np.uint8)]
    for _ in range(levels):
        pyr.append((cv2.pyrDown(pyr[-1] * 255) > 0).astype(np.uint8))
        
    merged = np.zeros_like(img, dtype=np.uint8)
    for level, bi in enumerate(pyr):
        if HAS_SKIMAGE:
            sk = skeletonize(bi > 0).astype(np.uint8)
        else:
            sk = zhang_suen_thinning(bi > 0)
        for _ in range(level): 
            sk = cv2.pyrUp(sk)
        sk = cv2.resize(sk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        merged = np.logical_or(merged, sk).astype(np.uint8)
    return zhang_suen_thinning(merged)

def quadtree_skeleton(img, min_size=8, occ_thresh=0.15):
    """基于四叉树分解的目标表达与骨架"""
    h, w = img.shape
    leaves = []
    
    def split(x, y, size):
        """递归将图像区域按四叉树规则分割，满足终止条件的叶节点加入列表。"""
        patch = img[y:y+size, x:x+size]
        if patch.size == 0: 
            return
        occ = np.mean(patch > 0)
        # 满足不可分割条件：尺寸太小 或 全空或全满
        if size <= min_size or occ < occ_thresh or occ > 1 - occ_thresh:
            if occ > occ_thresh: leaves.append((x, y, size))
            return
        half = size // 2
        if half < 1:
            return
        split(x, y, half)
        split(x+half, y, half)
        split(x, y+half, half)
        split(x+half, y+half, half)
        
    size = 2**int(math.log2(min(h, w)))
    split(0, 0, size)
    
    canvas = np.zeros_like(img)
    def get_center(box):
        """计算四叉树叶节点 (x, y, size) 的中心坐标。"""
        return box[0] + box[2] // 2, box[1] + box[2] // 2

    for i, a in enumerate(leaves):
        for b in leaves[i+1:]:
            ax, ay, asz = a
            bx, by, bsz = b
            touch_x = (ax + asz == bx) or (bx + bsz == ax)
            overlap_y = not (ay + asz <= by or by + bsz <= ay)
            touch_y = (ay + asz == by) or (by + bsz == ay)
            overlap_x = not (ax + asz <= bx or bx + bsz <= ax)
            
            if (touch_x and overlap_y) or (touch_y and overlap_x):
                acx, acy = get_center(a)
                bcx, bcy = get_center(b)
                cv2.line(canvas, (acx, acy), (bcx, bcy), 1, 1)
                
        acx, acy = get_center(a)
        cv2.circle(canvas, (acx, acy), 1, 1, -1)
        
    return zhang_suen_thinning(canvas)

if __name__ == '__main__':
    img_path = local_path("img.png")
    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"Error: 无法加载图片 {img_path}")
        exit()
        
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    binary = to_binary(gray)

    if HAS_SKIMAGE:
        methods = {
            '原图二值化 (Binary)': binary,
            'Zhang-Suen 细化': zhang_suen_thinning(binary),
            '距离变换中轴 \n(skimage medial_axis)': medial_axis(binary).astype(np.uint8),
            '四叉树结构表示': quadtree_skeleton(binary),
            '金字塔多尺度骨架': pyramid_skeleton(binary),
            'OpenCV 距离变换骨架': opencv_dt_skeleton(binary),
            'skimage skeletonize': skeletonize(binary).astype(np.uint8)
        }
    else:
        methods = {
            '原图二值化 (Binary)': binary,
            'Zhang-Suen 细化': zhang_suen_thinning(binary),
            '距离变换中轴 \n(无skimage=Zhang)': zhang_suen_thinning(binary),
            '四叉树结构表示': quadtree_skeleton(binary),
            '金字塔多尺度骨架': pyramid_skeleton(binary),
            'OpenCV 距离变换骨架': opencv_dt_skeleton(binary),
            'skimage skeletonize\n(无skimage=Zhang)': zhang_suen_thinning(binary)
        }

    plt.figure(figsize=(16, 9))
    plt.suptitle("目标区域表达与描述：不同方法对比", fontsize=18)
    
    for i, (title, img_res) in enumerate(methods.items()):
        plt.subplot(2, 4, i+1)
        plt.title(title, fontsize=12)
        # 结果图通常黑色背景比较难看清全貌，可以将前背景稍微反转使得底色变白显示，视觉更清晰。若需要保持原图可用 img_res 显示即可。这里采用原图黑底白线的显示:
        plt.imshow(img_res, cmap='gray')
        plt.axis('off')
        
    plt.tight_layout()
    # 调整suptitle和图表间的距离
    plt.subplots_adjust(top=0.90) 
    plt.savefig(local_path("comparison_result.png"), dpi=300)
    print("对比图已保存为 comparison_result.png")
    plt.show()
