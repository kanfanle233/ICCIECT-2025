"""
教学示例：hough trans

- 功能：演示 基元检测 中与“hough trans”相关的核心流程。
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
def generate_test_image():
    """生成包含各种几何形状的测试图像"""
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255  # 白色背景

    # 1. 绘制直线
    cv2.line(img, (50, 50), (200, 150), (0, 0, 0), 2)
    cv2.line(img, (50, 200), (200, 300), (0, 0, 0), 2)
    cv2.line(img, (300, 50), (300, 200), (0, 0, 0), 2)
    cv2.line(img, (350, 50), (350, 200), (0, 0, 0), 2)

    # 2. 绘制正方形
    cv2.rectangle(img, (500, 50), (600, 150), (0, 0, 0), 2)
    cv2.rectangle(img, (500, 200), (580, 280), (0, 0, 0), 2)

    # 3. 绘制圆形 (半径分别为 40 和 30)
    # 注意：线条宽度为2，霍夫变换检测的是边缘，需要清晰的圆环
    cv2.circle(img, (150, 450), 40, (0, 0, 0), 2)
    cv2.circle(img, (300, 450), 30, (0, 0, 0), 2)

    # 4. 绘制椭圆
    cv2.ellipse(img, (550, 450), (60, 30), 0, 0, 360, (0, 0, 0), 2)
    cv2.ellipse(img, (700, 450), (40, 20), 45, 0, 360, (0, 0, 0), 2)

    return img


def detect_lines_and_parallel(img_gray):
    """使用霍夫概率变换检测直线，并找出平行线对。"""
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=10)

    result_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    parallel_pairs = []
    line_data = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = np.inf if x2 - x1 == 0 else (y2 - y1) / (x2 - x1)
            line_data.append({'line': (x1, y1, x2, y2), 'slope': slope})
            cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        slope_threshold = 0.15
        for i in range(len(line_data)):
            for j in range(i + 1, len(line_data)):
                s1, s2 = line_data[i]['slope'], line_data[j]['slope']
                is_parallel = (s1 == np.inf and s2 == np.inf) or \
                              (s1 != np.inf and s2 != np.inf and abs(s1 - s2) < slope_threshold)

                if is_parallel:
                    parallel_pairs.append((line_data[i]['line'], line_data[j]['line']))

        for l1, l2 in parallel_pairs:
            cv2.line(result_img, (l1[0], l1[1]), (l1[2], l1[3]), (255, 0, 0), 3)
            cv2.line(result_img, (l2[0], l2[1]), (l2[2], l2[3]), (255, 0, 0), 3)

    return result_img, len(parallel_pairs)


def detect_squares(img_gray):
    """通过轮廓检测和多边形近似识别正方形。"""
    edges = cv2.Canny(img_gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    count = 0

    for cnt in contours:
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4 and cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(approx)
            if 0.8 <= float(w) / h <= 1.2 and cv2.isContourConvex(approx):
                cv2.drawContours(result_img, [approx], -1, (0, 0, 255), 3)
                count += 1
    return result_img, count


def detect_circles(img_gray):
    """使用霍夫梯度法检测圆形，返回标注结果图和检测到的圆数量。

    优化后的圆形检测
    关键调整：
    1. dp=1: 累加器分辨率与图像一致
    2. minDist=20: 圆心最小距离，设小一点防止漏检靠近的圆
    3. param1=50: Canny高阈值，保持默认或稍低
    4. param2=20: 【关键】累加器阈值。原代码40可能太高，导致投票不足。降低此值可提高灵敏度。
    5. minRadius=20, maxRadius=50: 确保覆盖生成的 30 和 40 半径
    """
    # 确保图像是 uint8 且是单通道
    if img_gray.dtype != np.uint8:
        img_gray = np.uint8(img_gray)

    circles = cv2.HoughCircles(
        img_gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,  # 减小最小距离
        param1=50,  # Canny 高阈值
        param2=20,  # 【重要】降低阈值，让检测更灵敏
        minRadius=20,  # 覆盖半径 30
        maxRadius=50  # 覆盖半径 40
    )

    result_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    count = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            # 绘制圆周 (青色)
            cv2.circle(result_img, center, radius, (0, 255, 255), 3)
            # 绘制圆心 (红色)
            cv2.circle(result_img, center, 2, (0, 0, 255), 3)
            count += 1
    else:
        print("未检测到任何圆形。尝试进一步降低 param2 或检查图像边缘。")

    return result_img, count


def detect_ellipses(img_gray):
    """通过轮廓拟合椭圆，过滤过圆或过小的结果。"""
    edges = cv2.Canny(img_gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    count = 0

    for cnt in contours:
        if len(cnt) >= 5:
            try:
                ellipse = cv2.fitEllipse(cnt)
                (center, axes, angle) = ellipse
                ratio = min(axes) / max(axes)
                if max(axes) > 20 and 0.4 <= ratio < 0.85:
                    cv2.ellipse(result_img, ellipse, (255, 0, 255), 3)
                    count += 1
            except cv2.error:
                continue
    return result_img, count


# --- 主程序 ---
original_img = generate_test_image()

# 保存生成的图片
cv2.imwrite(local_path("hough.jpg"), original_img)
print(f"图像已保存为 {local_path('hough.jpg')}")

gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

# 执行检测
img_lines, n_lines = detect_lines_and_parallel(gray_img)
img_squares, n_squares = detect_squares(gray_img)
img_circles, n_circles = detect_circles(gray_img)
img_ellipses, n_ellipses = detect_ellipses(gray_img)

print(f"检测结果统计:")
print(f"- 平行线对: {n_lines}")
print(f"- 正方形: {n_squares}")
print(f"- 圆形: {n_circles}")
print(f"- 椭圆: {n_ellipses}")

# 准备绘图数据
plots = [
    (original_img, "Original Generated Image"),
    (img_lines, f"Line & Parallel Detection (Pairs: {n_lines})"),
    (img_squares, f"Square Detection (Count: {n_squares})"),
    (img_circles, f"Circle Detection (Count: {n_circles})"),
    (img_ellipses, f"Ellipse Detection (Count: {n_ellipses})")
]

plt.figure(figsize=(20, 12))
for i, (img, title) in enumerate(plots):
    plt.subplot(2, 3, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title, fontsize=14)
    plt.axis('off')

plt.subplot(2, 3, 6)
plt.text(0.1, 0.8, f'Detection Summary:', fontsize=16, fontweight='bold')
plt.text(0.1, 0.6, f'Parallel Line Pairs: {n_lines}', fontsize=14)
plt.text(0.1, 0.5, f'Squares: {n_squares}', fontsize=14)
plt.text(0.1, 0.4, f'Circles: {n_circles}', fontsize=14)
plt.text(0.1, 0.3, f'Ellipses: {n_ellipses}', fontsize=14)
plt.axis('off')

plt.tight_layout()
plt.show()
