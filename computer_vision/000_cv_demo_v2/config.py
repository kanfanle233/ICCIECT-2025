"""
000_cv_demo_v2 的统一配置中心。

面向零基础同学时，最容易混乱的是“模型文件、中文标签、运行设备”散落在各处。
这里把它们集中到一个文件中，便于按主题理解：
1. 任务字典 `TASK_INFO` 说明“每个按钮对应什么模型、输出什么结果”。
2. 标签字典 `COCO_ZH` / `IMAGENET_ZH` 负责把英文类别翻译成中文。
3. 设备信息 `DEVICE_INFO` 负责统一决定本次推理应该优先走 MPS、CUDA 还是 CPU。
"""

import os
from pathlib import Path


def _build_device_info(device: str, reason: str) -> dict:
    """把设备选择结果整理成字典，方便界面层直接显示。"""
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
    """检测可用计算设备，顺序固定为 MPS -> CUDA -> CPU。"""
    override = os.environ.get("CV_DEVICE", "").strip().lower()
    if override in ("mps", "cuda", "cpu"):
        return _build_device_info(
            override,
            f"检测到环境变量 CV_DEVICE={override}，因此优先尊重人工指定的设备。",
        )

    try:
        import torch
    except ImportError:
        return _build_device_info(
            "cpu",
            "当前解释器没有安装 PyTorch，无法检测 GPU，因此保守回退到 CPU。",
        )

    if torch.backends.mps.is_available():
        return _build_device_info(
            "mps",
            "这是本地 Mac 可直接使用的 Metal/MPS 加速器，所以按约定放在第一优先级。",
        )

    if torch.cuda.is_available():
        return _build_device_info(
            "cuda",
            "当前机器没有可用的 MPS，但检测到了 NVIDIA CUDA，因此改用 CUDA 推理。",
        )

    if torch.backends.mps.is_built():
        reason = "PyTorch 已编译 MPS 支持，但当前系统没有开放可用的 MPS 设备，因此回退到 CPU。"
    else:
        reason = "当前解释器既没有可用的 MPS，也没有可用的 CUDA，所以只能使用 CPU。"
    return _build_device_info("cpu", reason)


def detect_device():
    """只返回设备字符串，适合直接传给 YOLO 或日志。"""
    return detect_device_info()["device"]


DEVICE_INFO = detect_device_info()
DEVICE = DEVICE_INFO["device"]
DEVICE_DISPLAY = DEVICE_INFO["display"]
DEVICE_REASON = DEVICE_INFO["reason"]

# ========== 路径配置 ==========
SAMPLES_DIR = Path(__file__).parent / "samples"
IMAGES_DIR = SAMPLES_DIR / "images"
VIDEOS_DIR = SAMPLES_DIR / "videos"

# ========== COCO 类别中文映射 ==========
COCO_ZH = {
    "person": "人",
    "bicycle": "自行车",
    "car": "汽车",
    "motorcycle": "摩托车",
    "airplane": "飞机",
    "bus": "公交车",
    "train": "火车",
    "truck": "卡车",
    "boat": "船",
    "traffic light": "红绿灯",
    "fire hydrant": "消防栓",
    "stop sign": "停车标志",
    "parking meter": "停车计时器",
    "bench": "长椅",
    "bird": "鸟",
    "cat": "猫",
    "dog": "狗",
    "horse": "马",
    "sheep": "羊",
    "cow": "牛",
    "elephant": "大象",
    "bear": "熊",
    "zebra": "斑马",
    "giraffe": "长颈鹿",
    "backpack": "背包",
    "umbrella": "雨伞",
    "handbag": "手提包",
    "tie": "领带",
    "suitcase": "行李箱",
    "frisbee": "飞盘",
    "skis": "滑雪板",
    "snowboard": "滑雪板",
    "sports ball": "球",
    "kite": "风筝",
    "baseball bat": "棒球棒",
    "baseball glove": "棒球手套",
    "skateboard": "滑板",
    "surfboard": "冲浪板",
    "tennis racket": "网球拍",
    "bottle": "瓶子",
    "wine glass": "酒杯",
    "cup": "杯子",
    "fork": "叉子",
    "knife": "刀",
    "spoon": "勺子",
    "bowl": "碗",
    "banana": "香蕉",
    "apple": "苹果",
    "sandwich": "三明治",
    "orange": "橙子",
    "broccoli": "西兰花",
    "carrot": "胡萝卜",
    "hot dog": "热狗",
    "pizza": "披萨",
    "donut": "甜甜圈",
    "cake": "蛋糕",
    "chair": "椅子",
    "couch": "沙发",
    "potted plant": "盆栽",
    "bed": "床",
    "dining table": "餐桌",
    "toilet": "马桶",
    "tv": "电视",
    "laptop": "笔记本电脑",
    "mouse": "鼠标",
    "remote": "遥控器",
    "keyboard": "键盘",
    "cell phone": "手机",
    "microwave": "微波炉",
    "oven": "烤箱",
    "toaster": "烤面包机",
    "sink": "水槽",
    "refrigerator": "冰箱",
    "book": "书",
    "clock": "时钟",
    "vase": "花瓶",
    "scissors": "剪刀",
    "teddy bear": "泰迪熊",
    "hair drier": "吹风机",
    "toothbrush": "牙刷",
}

# ========== ImageNet 类别中文映射 ==========
IMAGENET_ZH = {
    "golden_retriever": "金毛犬",
    "tabby": "虎斑猫",
    "sports_car": "跑车",
    "passenger_car": "轿车",
    "airliner": "客机",
    "speedboat": "快艇",
    "traffic_light": "红绿灯",
    "fire_engine": "消防车",
    "ambulance": "救护车",
    "school_bus": "校车",
    "minibus": "面包车",
    "moving_van": "货车",
    "police_van": "警车",
    "recreational_vehicle": "房车",
    "limousine": "豪华轿车",
    "convertible": "敞篷车",
    "mountain_bike": "山地车",
    "ice_bear": "北极熊",
    "giant_panda": "大熊猫",
    "macaw": "金刚鹦鹉",
    "hummingbird": "蜂鸟",
    "pelican": "鹈鹕",
    "king_penguin": "帝企鹅",
    "jellyfish": "水母",
    "starfish": "海星",
    "broccoli": "西兰花",
    "cauliflower": "花椰菜",
    "zucchini": "西葫芦",
    "cucumber": "黄瓜",
    "mushroom": "蘑菇",
    "Granny_Smith": "青苹果",
    "strawberry": "草莓",
    "orange": "橙子",
    "lemon": "柠檬",
    "pineapple": "菠萝",
    "banana": "香蕉",
    "pomegranate": "石榴",
    "pizza": "披萨",
    "hotpot": "火锅",
}

# ========== 任务配置 ==========
TASK_INFO = {
    "detection": {
        "name": "目标检测",
        "desc": "识别物体并用边界框定位，输出类别 + 坐标",
        "model": "yolo11n.pt",
    },
    "segmentation": {
        "name": "图像分割",
        "desc": "像素级精确划分物体区域，输出掩膜",
        "model": "yolo11n-seg.pt",
    },
    "classification": {
        "name": "图像分类",
        "desc": "判断整张图像所属类别",
        "model": "yolo11n-cls.pt",
    },
    "pose": {
        "name": "姿态估计",
        "desc": "检测人体 17 个关键点",
        "model": "yolo11n-pose.pt",
    },
    "obb": {
        "name": "定向检测",
        "desc": "检测旋转目标，输出带角度边界框",
        "model": "yolo11n-obb.pt",
    },
}

# ========== 绘图调色板 ==========
PALETTE = [
    (79, 78, 200),
    (78, 77, 26),
    (139, 116, 100),
    (114, 138, 91),
    (94, 94, 139),
    (87, 168, 227),
    (140, 123, 107),
    (74, 107, 155),
    (90, 107, 74),
]
