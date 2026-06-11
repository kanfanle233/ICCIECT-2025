"""
tag_01 的统一配置文件。

这个脚本组是一个“多模式实时演示”项目，所以这里集中保存三类最重要的信息：
1. 模型和视频路径，避免路径散落在多个脚本里。
2. 阈值、关键点索引、轨迹长度等教学参数，方便学生只改一处就能观察效果。
3. 设备检测逻辑，统一按照 MPS -> CUDA -> CPU 的顺序选择后端。
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONF_THRESH = 0.45
IOU_THRESH = 0.45

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

MODEL_DETECT = os.path.join(BASE_DIR, "yolo11n.pt")
MODEL_POSE = os.path.join(BASE_DIR, "yolo11n-pose.pt")
MODEL_SEG = os.path.join(BASE_DIR, "yolo11n-seg.pt")

MODE_MODELS = {
    "detection": MODEL_DETECT,
    "tracking": MODEL_DETECT,
    "pose": MODEL_POSE,
    "segmentation": MODEL_SEG,
}
# 这里用字典而不是 if/else 链，是因为“模式名 -> 模型路径”本身就是天然映射关系，
# 初学者调试时也能一眼看出 tracking 和 detection 共用同一个检测模型。

USE_TRACKER = "bytetrack.yaml"

POSE_CONF_THRESH = 0.5
POSE_IOU_THRESH = 0.3

ACTION_KEYPOINT_INDICES = {
    "nose": 0,
    "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

MIN_KEYPOINTS_FOR_SKELETON = 5

T_POSE_THRESH = 30
SQUAT_KNEE_HIP_HORIZONTAL_THRESH = 40
SQUAT_HIP_ANKLE_VERTICAL_THRESH = 120
RAISE_HAND_THRESH = 20

MASK_THRESH = 0.5
MASK_ALPHA = 0.35

TRAJECTORY_MAX_LEN = 45
ENTITY_PRINT_EVERY = 15

VIDEO_PATH = os.path.join(BASE_DIR, "dancetrack0040.mp4")

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128),
    (128, 128, 0), (128, 0, 128), (0, 128, 128),
]

SKELETON_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
]


def _build_device_info(device, reason):
    """整理设备展示文本，避免 main.py 和工具脚本重复拼接说明。"""
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
    """按 MPS -> CUDA -> CPU 的顺序检测推理后端。"""
    override = os.environ.get("CV_DEVICE", "").strip().lower()
    if override in {"mps", "cuda", "cpu"}:
        return _build_device_info(
            override,
            f"检测到环境变量 CV_DEVICE={override}，因此优先使用人工指定的设备。",
        )

    try:
        import torch
    except ImportError:
        return _build_device_info(
            "cpu",
            "当前解释器没有安装 PyTorch，无法检测 GPU，所以直接回退到 CPU。",
        )

    if torch.backends.mps.is_available():
        return _build_device_info(
            "mps",
            "当前是本地 Mac 环境，MPS 是最贴近本机的硬件加速器，因此放在第一优先级。",
        )

    if torch.cuda.is_available():
        return _build_device_info(
            "cuda",
            "没有可用的 MPS，但检测到了 NVIDIA CUDA，因此改用 CUDA 推理。",
        )

    if torch.backends.mps.is_built():
        reason = "PyTorch 已编译 MPS 支持，但当前系统没有开放可用的 MPS 设备，所以保底回退到 CPU。"
    else:
        reason = "当前机器既没有可用的 MPS，也没有可用的 CUDA，因此只能使用 CPU。"
    return _build_device_info("cpu", reason)


def detect_device():
    """只返回设备字符串，适合直接传给 YOLO。"""
    return detect_device_info()["device"]


def ensure_models_exist():
    """启动前检查模型文件是否存在，避免运行到中途才报错。"""
    for name, path in [
        ("detection/tracking", MODEL_DETECT),
        ("pose", MODEL_POSE),
        ("segmentation", MODEL_SEG),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"模型文件未找到: {path}\n"
                f"请确保 {name} 模型已下载到项目根目录"
            )
