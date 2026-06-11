"""
独立的实体统计脚本。

它和主程序不同，不负责把结果画在屏幕上，而是专注于把“人 / 动物 / 物品”三类目标统计出来。
这样拆开后，初学者可以单独运行这个脚本，先理解检测结果的数据结构，再回到主循环看实时界面。
"""

import argparse
import os
import sys
from typing import Dict

import cv2
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


ANIMAL_LABELS = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}


def categorize_label(label: str) -> str:
    """将检测标签归类为 'person'、'animal' 或 'object'。"""
    normalized = label.strip().lower()
    if normalized == "person":
        return "person"
    if normalized in ANIMAL_LABELS:
        return "animal"
    return "object"


def summarize_result(result, names: Dict[int, str], conf_thresh: float) -> Dict[str, Dict[str, int]]:
    """把单帧检测结果整理成三层字典。

    返回结构示例：
    {
        "person": {"person": 2},
        "animal": {"dog": 1},
        "object": {"chair": 3}
    }
    用字典嵌套字典的好处是：类别分组和同类计数可以同时表达。
    """
    summary = {"person": {}, "animal": {}, "object": {}}
    if result.boxes is None:
        return summary

    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < conf_thresh:
            continue

        cls_id = int(box.cls[0])
        label = names.get(cls_id, str(cls_id))
        category = categorize_label(label)
        summary[category][label] = summary[category].get(label, 0) + 1

    return summary


def summary_to_text(summary: Dict[str, Dict[str, int]]) -> str:
    """将实体统计字典格式化为可读的文本字符串（如 '人: 2 | 动物: 狗 x1 | 物品: 无'）。"""
    sections = []
    title_map = {
        "person": "人",
        "animal": "动物",
        "object": "物品",
    }

    for key in ("person", "animal", "object"):
        items = summary[key]
        if not items:
            sections.append(f"{title_map[key]}: 无")
            continue
        detail = ", ".join(f"{name} x{count}" for name, count in sorted(items.items()))
        sections.append(f"{title_map[key]}: {detail}")

    return " | ".join(sections)


def open_capture(source: str, video_path: str, camera_id: int):
    """根据输入源类型打开摄像头或本地视频，返回捕获对象和标签。"""
    if source == "camera":
        cap = cv2.VideoCapture(camera_id)
        label = f"摄像头({camera_id})"
    else:
        cap = cv2.VideoCapture(video_path)
        label = f"本地视频({video_path})"

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"无法打开输入源: {label}")
    return cap, label


def run_recognizer(source: str, video_path: str, camera_id: int, print_every: int, show: bool):
    """独立运行实体识别循环：打开输入源，逐帧检测并按指定间隔输出统计。"""
    device_info = config.detect_device_info()
    device = device_info["device"]
    model = YOLO(config.MODEL_DETECT).to(device)
    names = model.names

    cap, label = open_capture(source, video_path, camera_id)
    print(f"输入源: {label}")
    print(f"计算设备: {device_info['display']}")
    print(f"选择原因: {device_info['reason']}")
    print("开始识别（按 Q 退出）...")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            if source == "video":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("摄像头读取失败，程序结束。")
            break

        frame_idx += 1
        results = model(
            frame,
            conf=config.CONF_THRESH,
            iou=config.IOU_THRESH,
            device=device,
            verbose=False,
        )
        summary = summarize_result(results[0], names, conf_thresh=config.CONF_THRESH)

        if frame_idx % max(print_every, 1) == 0:
            print(summary_to_text(summary))

        if show:
            cv2.imshow("Entity Recognizer", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    if show:
        cv2.destroyAllWindows()


def parse_args():
    """解析命令行参数，返回输入源、视频路径、摄像头ID等配置。"""
    parser = argparse.ArgumentParser(description="识别并输出人/动物/物品三类目标。")
    parser.add_argument(
        "--source",
        choices=["camera", "video"],
        default="video",
        help="输入源类型: camera 或 video",
    )
    parser.add_argument(
        "--video-path",
        default=config.VIDEO_PATH,
        help="本地视频路径（source=video 时使用）",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="摄像头编号（source=camera 时使用）",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=15,
        help="每隔多少帧输出一次统计结果",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="是否显示视频窗口（默认不显示）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_recognizer(
        source=args.source,
        video_path=args.video_path,
        camera_id=args.camera_id,
        print_every=args.print_every,
        show=args.show,
    )
