"""跟踪模式：维护目标 ID、轨迹、速度和平滑统计信息。"""

import cv2
import math
import os
import sys
import time

try:
    from .. import config
    from ..utils.helpers import draw_detection_box, get_color, TrajectoryBuffer
except ImportError:
    # Fallback for running this file directly: python tracking.py
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import config
    from utils.helpers import draw_detection_box, get_color, TrajectoryBuffer


trajectory_buffer = TrajectoryBuffer()
track_motion_state = {}
unique_track_ids = set()

SPEED_EMA_ALPHA = 0.35
STATIONARY_SPEED_THRESHOLD = 8.0  # px/s


def reset_tracking_state():
    """清空轨迹缓存与速度状态，供主循环在重置时调用。"""
    global trajectory_buffer, unique_track_ids
    trajectory_buffer = TrajectoryBuffer()
    track_motion_state.clear()
    unique_track_ids = set()


def _update_track_speed(track_id, center, now_ts):
    """更新速度指数滑动平均值。

    `track_motion_state` 是一个字典：
    `{track_id: {"center": (x, y), "ts": 时间戳, "speed_ema": 平滑速度}}`
    之所以不用单独的三个列表，是因为字典按 ID 查找更直接，也更不容易把不同目标的状态串位。
    """
    prev_state = track_motion_state.get(track_id)
    if prev_state is None:
        track_motion_state[track_id] = {
            "center": center,
            "ts": now_ts,
            "speed_ema": 0.0,
        }
        return 0.0

    prev_center = prev_state["center"]
    prev_ts = prev_state["ts"]
    dt = max(now_ts - prev_ts, 1e-6)
    dist = math.hypot(center[0] - prev_center[0], center[1] - prev_center[1])
    inst_speed = dist / dt

    speed_ema = (
        prev_state["speed_ema"] * (1.0 - SPEED_EMA_ALPHA)
        + inst_speed * SPEED_EMA_ALPHA
    )

    track_motion_state[track_id] = {
        "center": center,
        "ts": now_ts,
        "speed_ema": speed_ema,
    }
    return speed_ema


def _draw_tracking_hud(frame, active_count, avg_speed, unique_count, fastest_id, fastest_speed):
    """Draw tracking statistics on the frame."""
    h, w = frame.shape[:2]
    x1 = max(w - 360, 10)
    y1 = 10
    x2 = w - 10
    y2 = min(y1 + 106, h - 10)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)

    lines = [
        f"Active IDs: {active_count}",
        f"Avg speed: {avg_speed:.1f} px/s",
        f"Unique IDs: {unique_count}",
    ]
    if fastest_id is None:
        lines.append("Fastest: N/A")
    else:
        lines.append(f"Fastest: ID{fastest_id} {fastest_speed:.1f} px/s")

    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (x1 + 10, y1 + 22 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
        )


def process_tracking(frame, model, names, device="cpu"):
    """对单帧执行多目标跟踪，绘制检测框、轨迹线和统计面板。"""
    global trajectory_buffer, unique_track_ids

    results = model.track(
        frame,
        conf=config.CONF_THRESH,
        iou=config.IOU_THRESH,
        persist=True,
        tracker=config.USE_TRACKER,
        device=device,
        verbose=False,
    )
    result = results[0]

    active_ids = set()
    speed_values = []
    fastest_id = None
    fastest_speed = 0.0
    now_ts = time.time()

    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()

        for i, track_id in enumerate(track_ids):
            active_ids.add(track_id)
            x1, y1, x2, y2 = boxes[i]
            cls_id = classes[i]
            conf = confs[i]
            label = names.get(cls_id, str(cls_id))
            color = get_color(track_id)

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            speed_ema = _update_track_speed(track_id, (cx, cy), now_ts)
            speed_values.append(speed_ema)
            if speed_ema > fastest_speed:
                fastest_id = track_id
                fastest_speed = speed_ema

            move_state = "MOVE" if speed_ema >= STATIONARY_SPEED_THRESHOLD else "IDLE"
            draw_detection_box(
                frame,
                (x1, y1, x2, y2),
                f"ID{track_id} {label} {speed_ema:.1f}px/s {move_state}",
                conf,
                color,
            )

            trajectory_buffer.update(track_id, (cx, cy))
            trajectory_buffer.draw(frame, track_id, color)

    unique_track_ids.update(active_ids)
    trajectory_buffer.clear_except(active_ids)
    for tid in list(track_motion_state.keys()):
        if tid not in active_ids:
            del track_motion_state[tid]

    avg_speed = sum(speed_values) / len(speed_values) if speed_values else 0.0
    _draw_tracking_hud(
        frame=frame,
        active_count=len(active_ids),
        avg_speed=avg_speed,
        unique_count=len(unique_track_ids),
        fastest_id=fastest_id,
        fastest_speed=fastest_speed,
    )

    return frame
