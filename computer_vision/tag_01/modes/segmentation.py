"""分割模式：把实例掩膜涂到原图上，并叠加类别框。"""

import cv2
import numpy as np
import os
import sys

try:
    from .. import config
    from ..utils.helpers import draw_detection_box, get_color
except ImportError:
    # Fallback for running this file directly: python segmentation.py
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import config
    from utils.helpers import draw_detection_box, get_color


def process_segmentation(frame, model, names, device="cpu"):
    """执行实例分割并叠加可视化结果。"""
    results = model(
        frame,
        conf=config.CONF_THRESH,
        iou=config.IOU_THRESH,
        device=device,
        verbose=False,
    )
    result = results[0]

    overlay = frame.copy()

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.int().cpu().tolist()

        h, w = frame.shape[:2]

        for i, mask in enumerate(masks):
            binary_mask = (mask > config.MASK_THRESH).astype(np.uint8)
            binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            cls_id = classes[i]
            color_bgr = get_color(cls_id)
            color_bgr_array = np.array(color_bgr, dtype=np.uint8)

            colored_mask = np.zeros_like(overlay)
            colored_mask[binary_mask == 1] = color_bgr_array
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, config.MASK_ALPHA, 0)

            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color_bgr, 1)

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = names.get(cls_id, str(cls_id))
            color = get_color(cls_id)
            draw_detection_box(overlay, (x1, y1, x2, y2), label, conf, color)

    frame[:] = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

    return frame
