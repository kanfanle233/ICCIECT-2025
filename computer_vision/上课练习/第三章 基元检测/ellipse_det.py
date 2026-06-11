"""
教学示例：ellipse det

- 功能：演示 基元检测 中与“ellipse det”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# 1. 生成单个【正】椭圆图像 (角度=0)
def create_single_ellipse():
    """生成一张包含单个正椭圆的白色背景测试图像。"""
    img = np.ones((500, 700, 3), dtype=np.uint8) * 255
    # 中心 (350, 250), 长轴 120, 短轴 60, 角度 0
    center, axes, angle = (350, 250), (120, 60), 0
    cv2.ellipse(img, center, axes, angle, 0, 360, (0, 0, 0), 2)
    return img, (center, axes, angle)


# 2. 获取边缘点和梯度
def get_edge_data(gray):
    """对灰度图执行Canny边缘检测，返回边缘点坐标和对应梯度方向。"""
    edges = cv2.Canny(gray, 50, 150)
    y, x = np.where(edges > 0)
    points = np.column_stack((x, y)).astype(float)

    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grads = [np.arctan2(dy[int(yi), int(xi)], dx[int(yi), int(xi)]) for xi, yi in zip(x, y)]
    return points, np.array(grads)


# 3. 直径二分法 (仅找中心和半径)
def detect_diameter(points, shape):
    """使用直径二分法（随机点对中点投票）估计椭圆中心和长短轴。"""
    votes = defaultdict(int)
    n = len(points)
    if n < 2: return None

    # 随机采样点对投票
    idxs = np.random.choice(n, 4000, replace=True)
    for i in range(0, 4000, 2):
        p1, p2 = points[idxs[i]], points[idxs[i + 1]]
        if np.linalg.norm(p1 - p2) < 20: continue  # 过滤太近的点
        mid = (p1 + p2) / 2
        gx, gy = int(mid[0] // 5), int(mid[1] // 5)  # 5x5 网格投票
        votes[(gx, gy)] += 1

    if not votes: return None
    # 取票数最高的作为中心
    (gx, gy), _ = max(votes.items(), key=lambda x: x[1])
    cx, cy = (gx + 0.5) * 5, (gy + 0.5) * 5

    # 估算长短轴半径
    dists = np.linalg.norm(points - [cx, cy], axis=1)
    major_r = np.percentile(dists, 95)  # 95% 分位数为长轴
    minor_r = np.percentile(dists, 5)  # 5% 分位数为短轴

    return {'center': (cx, cy), 'axes': (major_r, minor_r)}


# 4. 弦切线法 (仅找中心和半径)
def detect_chord(points, grads):
    """使用弦切线法（按梯度分组拟合中点线交点）估计椭圆中心和长短轴。"""
    groups = defaultdict(list)
    # 按梯度方向分组 (0~pi)
    for i, p in enumerate(points):
        a = grads[i]
        if a < 0: a += np.pi
        bin_idx = int(a / (np.pi / 8)) % 8  # 分 8 组
        groups[bin_idx].append(p)

    lines = []
    # 每组内找平行弦中点，拟合直线
    for pts in groups.values():
        if len(pts) < 10: continue
        mids = []
        idxs = np.random.choice(len(pts), min(100, len(pts)), replace=False)
        for k in range(0, len(idxs) - 1, 2):
            mids.append((pts[idxs[k]] + pts[idxs[k + 1]]) / 2)
        if len(mids) < 5: continue

        mids_np = np.array(mids)
        mean = np.mean(mids_np, axis=0)
        cov = np.cov(mids_np - mean, rowvar=False)
        try:
            _, vecs = np.linalg.eig(cov)
            lines.append({'pt': mean, 'dir': vecs[:, 1]})
        except:
            continue

    centers = []
    # 求直线交点
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            l1, l2 = lines[i], lines[j]
            A = np.column_stack((l1['dir'], -l2['dir']))
            if abs(np.linalg.det(A)) < 1e-5: continue
            try:
                t = np.linalg.solve(A, l2['pt'] - l1['pt'])
                centers.append(l1['pt'] + t[0] * l1['dir'])
            except:
                continue

    if not centers: return None
    c_est = np.mean(centers, axis=0)

    # 估算半径
    dists = np.linalg.norm(points - c_est, axis=1)
    major_r = np.percentile(dists, 95)
    minor_r = np.percentile(dists, 5)

    return {'center': (c_est[0], c_est[1]), 'axes': (major_r, minor_r)}


# 5. 执行与绘图
img, truth = create_single_ellipse()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
points, grads = get_edge_data(gray)

res_diam = detect_diameter(points, gray.shape)
res_chord = detect_chord(points, grads)


# 绘图辅助函数
def draw(img, res, color):
    """在图像上绘制检测到的椭圆轮廓和中心点。"""
    out = img.copy()
    if res:
        c, a = res['center'], res['axes']
        # 角度固定为 0 (正椭圆)
        cv2.ellipse(out, (int(c[0]), int(c[1])), (int(a[0]), int(a[1])), 0, 0, 360, color, 2)
        cv2.circle(out, (int(c[0]), int(c[1])), 5, (0, 255, 0), -1)
    return out


img_diam = draw(img, res_diam, (255, 0, 0))  # 红色结果
img_chord = draw(img, res_chord, (0, 255, 255))  # 青色结果

plt.figure(figsize=(15, 5))

# 图 1: 原图
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f"Original Image\nTrue: Center{truth[0]}, Axes{truth[1]}")
plt.axis('off')

# 图 2: 直径二分法
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(img_diam, cv2.COLOR_BGR2RGB))
if res_diam:
    title_str = f"Diameter Bisection\nCenter: ({res_diam['center'][0]:.1f}, {res_diam['center'][1]:.1f})\nAxes: ({res_diam['axes'][0]:.1f}, {res_diam['axes'][1]:.1f})"
else:
    title_str = "Detection Failed"
plt.title(title_str)
plt.axis('off')

# 图 3: 弦切线法
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(img_chord, cv2.COLOR_BGR2RGB))
if res_chord:
    title_str = f"Chord-Tangent Method\nCenter: ({res_chord['center'][0]:.1f}, {res_chord['center'][1]:.1f})\nAxes: ({res_chord['axes'][0]:.1f}, {res_chord['axes'][1]:.1f})"
else:
    title_str = "Detection Failed"
plt.title(title_str)
plt.axis('off')

plt.tight_layout()
plt.show()

# 打印数值对比
print(f"真实值：中心={truth[0]}, 轴长={truth[1]}")
if res_diam:
    print(f"直径法：中心={res_diam['center']}, 轴长={res_diam['axes']}")
if res_chord:
    print(f"弦切法：中心={res_chord['center']}, 轴长={res_chord['axes']}")