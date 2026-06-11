"""
tag_01 的绘图与几何辅助函数。

这里的函数大多是“画框、画骨架、画状态栏”这类可视化工具。
为了让新手更容易看懂，和轨迹相关的状态都集中在 `TrajectoryBuffer` 里管理。
"""

from collections import defaultdict, deque

import cv2
import numpy as np
import config


class TrajectoryBuffer:
    """保存每个 track_id 最近若干帧的中心点轨迹。

    `tracks` 的数据结构是 `dict[int, deque[tuple]]`：
    - key: 跟踪器分配的目标编号。
    - value: 这个目标最近几帧的中心点。
    这里选 deque 的原因是“自动丢掉最旧的数据”比手动 pop(0) 更适合滑动窗口。
    """

    def __init__(self, max_len=None):
        if max_len is None:
            max_len = config.TRAJECTORY_MAX_LEN
        self.max_len = max_len
        self.tracks = defaultdict(lambda: deque(maxlen=self.max_len))

    def update(self, track_id, center):
        """追加一个目标的最新中心点坐标。"""
        self.tracks[track_id].append(center)

    def get(self, track_id):
        """返回指定目标的轨迹点列表。"""
        return list(self.tracks.get(track_id, []))

    def draw(self, frame, track_id, color):
        """在帧上绘制指定目标的轨迹线和轨迹点。"""
        points = self.tracks.get(track_id, [])
        if len(points) < 2:
            return
        for i in range(1, len(points)):
            pt1 = tuple(map(int, points[i - 1]))
            pt2 = tuple(map(int, points[i]))
            alpha = 0.3 + 0.7 * (i / len(points))
            thickness = max(1, int(3 * (i / len(points))))
            cv2.line(frame, pt1, pt2, color, thickness)
            cv2.circle(frame, pt1, 2, color, -1)

    def clear_except(self, active_ids):
        """清除所有不在活跃ID集合中的轨迹记录。"""
        stale = [tid for tid in self.tracks if tid not in active_ids]
        for tid in stale:
            del self.tracks[tid]


def calculate_angle(a, b, c):
    """计算以b为顶点的两条线段 ba 和 bc 之间的夹角（单位：度）。"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def get_color(idx, colors=None):
    """根据索引从颜色表中循环选取颜色。"""
    if colors is None:
        colors = config.COLORS
    return colors[idx % len(colors)]


def draw_detection_box(frame, box, label, conf, color):
    """在帧上绘制带标签和置信度的检测框。"""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_skeleton(frame, keypoints, conf_thresh=None):
    """在帧上绘制人体骨架的关键点和连接线。"""
    if conf_thresh is None:
        conf_thresh = config.POSE_CONF_THRESH

    points = []
    for kp in keypoints:
        x, y, c = int(kp[0]), int(kp[1]), float(kp[2])
        if c >= conf_thresh:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            points.append((x, y))
        else:
            points.append(None)

    for i, j in config.SKELETON_EDGES:
        if points[i] is not None and points[j] is not None:
            cv2.line(frame, points[i], points[j], (0, 255, 255), 2)

    return points


def draw_fps(frame, fps):
    """在帧左上角绘制FPS信息。"""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def draw_mode_indicator(frame, mode_name, color=(255, 255, 255)):
    """在帧右下角绘制当前模式名称指示器。"""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (w - 220, h - 40), (w - 10, h - 10), (0, 0, 0), -1)
    cv2.putText(frame, f"Mode: {mode_name}", (w - 210, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_device_info(frame, device, device_display):
    """在帧左上角绘制计算设备信息（带颜色指示灯）。"""
    icon_color = {"mps": (0, 255, 128), "cuda": (0, 255, 0), "cpu": (100, 100, 255)}.get(device, (255, 255, 255))
    cv2.circle(frame, (30, 50), 7, icon_color, -1)
    cv2.putText(frame, device_display, (45, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, icon_color, 1)


def draw_help_text(frame):
    """在帧左上角绘制键盘操作提示文本。"""
    h = frame.shape[0]
    hints = [
        "1:Detection  2:Tracking  3:Pose  4:Segmentation",
        "Q:Quit  R:Reset",
    ]
    y0 = 80
    for i, hint in enumerate(hints):
        cv2.putText(frame, hint, (10, y0 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
