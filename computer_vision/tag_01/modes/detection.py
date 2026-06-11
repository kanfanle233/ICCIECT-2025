"""目标检测模式：负责把 YOLO 的检测框画到当前帧上。"""

import cv2
import os
import sys

try:
    from .. import config
    from ..utils.helpers import draw_detection_box, get_color
except ImportError:
    # Fallback for running this file directly: python detection.py
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import config
    from utils.helpers import draw_detection_box, get_color


def process_detection(frame, model, names, device="cpu"):
    """执行单帧检测。

    `names` 是类别编号到类别名的字典，
    这样在绘制框时就能把模型内部的整数类别翻译成更容易理解的文字标签。
    """
    results = model(frame, conf=config.CONF_THRESH, iou=config.IOU_THRESH, device=device, verbose=False)
    result = results[0]

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = names.get(cls_id, str(cls_id))
            color = get_color(cls_id)
            draw_detection_box(frame, (x1, y1, x2, y2), label, conf, color)

    return frame
