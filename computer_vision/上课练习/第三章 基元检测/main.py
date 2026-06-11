"""
教学入口：第三章基元检测练习导航。

- 功能：告诉初学者这个目录下每个脚本大致演示什么。
- 主要数据结构：使用列表保存脚本说明，顺序打印。
- 这样设置的原因：原文件只是 PyCharm 默认模板，替换成导航脚本后更适合课堂使用。
"""

DEMOS = [
    ("edge_detection.py", "一阶/二阶边缘检测对比"),
    ("Harris_corner.py", "Harris 角点检测"),
    ("hough_trans.py", "霍夫变换检测线、圆和椭圆"),
    ("ellipse_det.py", "两种椭圆中心估计方法"),
    ("position_histogram_track.py", "位置直方图跟踪思想"),
    ("susan_edge.py", "SUSAN 边缘检测"),
]


def main():
    """打印本目录下各脚本的文件名和功能说明导航。"""
    print("第三章 基元检测练习导航")
    print("-" * 40)
    for file_name, description in DEMOS:
        print(f"{file_name:<30} {description}")
    print("\n建议：先运行 edge_detection.py，再看 Harris_corner.py 和 hough_trans.py。")


if __name__ == "__main__":
    main()
