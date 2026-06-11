"""
车辆计数教学脚本。

这个示例围绕一个最核心的问题展开：怎样判断“同一辆车是否穿过了统计线”。
因此代码刻意把三个部分拆开：
1. 设备检测：说明本次推理走 MPS、CUDA 还是 CPU。
2. 常量配置：把类别编号、跳帧比例、统计线位置放在顶部，便于课堂演示时调参。
3. 状态字典：用 `track_history` 保存每个车辆上一帧的纵坐标，避免重复计数。
"""

from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "yolov8n.pt"
VIDEO_PATH = PROJECT_DIR / "4.mp4"

# 这些类别编号来自 COCO 数据集：
# 2=car, 3=motorcycle, 5=bus, 7=truck。
ALLOWED_CLASS_IDS = [2, 3, 5, 7]
FRAME_SKIP = 5
VIRTUAL_LINE_RATIO = 0.6
DISPLAY_SIZE = (960, 540)


def _build_device_info(device, reason):
    """把设备与原因打包成字典，方便启动时直接打印。"""
    display_map = {
        "mps": "Apple MPS / Metal GPU Acceleration",
        "cuda": "NVIDIA CUDA GPU",
        "cpu": "CPU",
    }
    return {
        "device": device,
        "display": display_map.get(device, device.upper()),
        "reason": reason,
    }


def detect_device_info():
    """按本地 Mac 加速 -> CUDA -> CPU 的顺序检测设备。"""
    try:
        import torch
    except ImportError:
        return _build_device_info(
            "cpu",
            "当前解释器没有安装 PyTorch，无法检测 GPU，因此只能回退到 CPU。",
        )

    if torch.backends.mps.is_available():
        return _build_device_info(
            "mps",
            "检测到本机可用的 Apple Metal/MPS，因此优先走本地 Mac 加速。",
        )

    if torch.cuda.is_available():
        return _build_device_info(
            "cuda",
            "没有可用的 MPS，但检测到了 NVIDIA CUDA，因此改用 CUDA 推理。",
        )

    if torch.backends.mps.is_built():
        reason = "PyTorch 支持 MPS，但当前机器没有开放可用的 MPS 设备，所以保底回退到 CPU。"
    else:
        reason = "当前机器既没有可用的 MPS，也没有可用的 CUDA，因此只能使用 CPU。"
    return _build_device_info("cpu", reason)


def load_model(device):
    """加载 YOLO 模型，并显式放到统一选择好的设备上。"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型文件: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH)).to(device)


def open_video():
    """打开项目内置视频，保证脚本不依赖当前工作目录。"""
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"找不到视频文件: {VIDEO_PATH}")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {VIDEO_PATH}")
    return cap


def main():
    """启动车辆计数系统：加载模型，读取视频，通过虚拟线统计经过的车辆数量。"""
    device_info = detect_device_info()
    device = device_info["device"]
    print(f"计算设备: {device_info['display']}")
    print(f"选择原因: {device_info['reason']}")

    model = load_model(device)
    cap = open_video()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = int(height * VIRTUAL_LINE_RATIO)

    total_count = 0
    # `track_history` 的结构是 `{track_id: 上一帧中心点的 y 坐标}`。
    # 只保存上一帧就够用了，因为撞线只需要比较“上一次在不在线上方”和“这一次是否到了线下方”。
    track_history = {}
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("视频播放结束。")
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        results = model.track(
            frame,
            persist=True,
            classes=ALLOWED_CLASS_IDS,
            device=device,
            verbose=False,
        )

        result = results[0]
        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)

        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                prev_cy = track_history.get(track_id)
                if prev_cy is not None and prev_cy < line_y <= cy:
                    total_count += 1
                track_history[track_id] = cy

        annotated_frame = result.plot()
        cv2.line(annotated_frame, (0, line_y), (width, line_y), (0, 0, 255), 3)
        cv2.putText(
            annotated_frame,
            f"Traffic Count: {total_count}",
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 0, 0),
            4,
        )

        display_frame = cv2.resize(annotated_frame, DISPLAY_SIZE)
        cv2.imshow("Traffic Counting System", display_frame)

        if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
