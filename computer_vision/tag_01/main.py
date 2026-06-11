"""
tag_01 的主入口。

这个文件做三件事：
1. 选择输入源（摄像头或本地视频）。
2. 按统一设备规则加载模型并驱动四种视觉模式。
3. 维护实时循环里的公共状态，例如 FPS、实体摘要和当前模式编号。
"""

import time

import cv2
from ultralytics import YOLO

import config
from utils.helpers import draw_fps, draw_mode_indicator, draw_help_text, draw_device_info
from utils.entity_recognizer import summarize_result, summary_to_text
from modes.detection import process_detection
from modes.tracking import process_tracking
from modes.tracking import reset_tracking_state
from modes.pose import process_pose
from modes.segmentation import process_segmentation


def load_models(device):
    """加载演示所需的全部模型，并统一移动到目标设备。"""
    config.ensure_models_exist()
    return {
        mode_name: YOLO(model_path).to(device)
        for mode_name, model_path in config.MODE_MODELS.items()
    }


def select_input_source():
    """Ask user to select camera or local video source."""
    print("请选择输入源：")
    print("[1] 摄像头")
    print(f"[2] 本地视频（默认：{config.VIDEO_PATH}）")

    while True:
        try:
            choice = input("请输入 1 或 2（回车默认 2）: ").strip()
        except EOFError:
            choice = "2"
        if choice == "":
            choice = "2"

        if choice == "1":
            return {"source": "camera", "value": 0, "label": "摄像头(0)"}

        if choice == "2":
            try:
                video_path = input(
                    f"请输入本地视频路径（回车使用默认：{config.VIDEO_PATH}）: "
                ).strip()
            except EOFError:
                video_path = ""
            video_path = video_path or config.VIDEO_PATH
            return {"source": "video", "value": video_path, "label": f"本地视频: {video_path}"}

        print("输入无效，请输入 1 或 2。")


def get_video_source(source_info):
    """Open the selected source and return capture object."""
    is_camera = source_info["source"] == "camera"
    source_value = source_info["value"]
    cap = cv2.VideoCapture(source_value)
    if cap.isOpened():
        return cap, is_camera, source_info["label"]

    cap.release()
    if is_camera:
        raise RuntimeError("无法打开摄像头，请检查摄像头权限或设备占用。")
    raise RuntimeError(f"无法打开本地视频: {source_value}")


def draw_entity_summary(frame, summary_text):
    """把实体统计绘制到画面下方。

    OpenCV 原生文字渲染对中文不稳定，所以这里故意显示 ASCII 文本，
    终端里仍然打印中文，兼顾教学可读性和窗口稳定性。
    """
    if not summary_text:
        return

    max_chars = 80
    display_text = summary_text if len(summary_text) <= max_chars else summary_text[: max_chars - 3] + "..."
    y = frame.shape[0] - 58
    cv2.rectangle(frame, (10, y - 20), (min(frame.shape[1] - 10, 980), y + 6), (0, 0, 0), -1)
    cv2.putText(frame, display_text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)


def to_ascii_overlay_text(summary_text):
    """Convert Chinese summary labels to ASCII for stable OpenCV text rendering."""
    replacements = {
        "人:": "Person:",
        "动物:": "Animal:",
        "物品:": "Object:",
        "无": "None",
    }
    converted = summary_text
    for src, dst in replacements.items():
        converted = converted.replace(src, dst)
    return converted


def main():
    """启动YOLO实时推理演示：加载模型、选择输入源，按键盘切换四种视觉模式。"""
    print("=" * 55)
    print("  计算机视觉 YOLO 实时推理演示实验室")
    print("  展示: 检测 -> 跟踪 -> 姿态 -> 分割 的完整过程")
    print("=" * 55)
    print()
    print("[1] 目标检测    - 画面里有什么、在哪里")
    print("[2] 多目标跟踪  - 是不是刚才那个、往哪走了")
    print("[3] 姿态估计    - 关节点在哪里、动作是什么")
    print("[4] 实例分割    - 精确轮廓在哪里")
    print("[Q] 退出")
    print()

    device_info = config.detect_device_info()
    device = device_info["device"]
    device_display = device_info["display"]
    print(f"计算设备: {device_display}")
    print(f"选择原因: {device_info['reason']}")
    print()

    models = load_models(device)
    source_info = select_input_source()
    cap, is_camera, source_label = get_video_source(source_info)
    print(f"输入源: {source_label}")
    print()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    names_raw = models["detection"].names
    if isinstance(names_raw, list):
        names = {idx: name for idx, name in enumerate(names_raw)}
    else:
        names = names_raw

    # 这里使用“编号 -> (模式名, 处理函数)”的字典，
    # 是因为按键数字本身就是索引，映射结构比一连串 if/elif 更适合教学展示。
    mode_processors = {
        1: ("Detection", lambda f: process_detection(f, models["detection"], names, device)),
        2: ("Tracking", lambda f: process_tracking(f, models["tracking"], names, device)),
        3: ("Pose", lambda f: process_pose(f, models["pose"], device)),
        4: ("Segmentation", lambda f: process_segmentation(f, models["segmentation"], names, device)),
    }

    current_mode = 1
    prev_time = time.time()
    fps = 0.0
    frame_idx = 0
    entity_print_every = config.ENTITY_PRINT_EVERY
    latest_entity_console_text = "人: 无 | 动物: 无 | 物品: 无"
    latest_entity_overlay_text = to_ascii_overlay_text(latest_entity_console_text)

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_camera:
                print("摄像头画面读取失败，程序结束。")
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        frame_idx += 1

        if frame_idx % entity_print_every == 0:
            entity_results = models["detection"](
                frame,
                conf=config.CONF_THRESH,
                iou=config.IOU_THRESH,
                device=device,
                verbose=False,
            )
            entity_summary = summarize_result(entity_results[0], names, conf_thresh=config.CONF_THRESH)
            latest_entity_console_text = summary_to_text(entity_summary)
            latest_entity_overlay_text = to_ascii_overlay_text(latest_entity_console_text)
            print(latest_entity_console_text)

        mode_name, processor = mode_processors[current_mode]
        frame = processor(frame)

        curr_time = time.time()
        elapsed = curr_time - prev_time
        prev_time = curr_time
        fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if elapsed > 0 else fps

        draw_fps(frame, fps)
        draw_device_info(frame, device, device_display)
        draw_mode_indicator(frame, mode_name)
        draw_help_text(frame)
        draw_entity_summary(frame, latest_entity_overlay_text)

        cv2.imshow("YOLO Vision Lab", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

        if key in (ord("r"), ord("R")):
            if not is_camera:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            latest_entity_console_text = "人: 无 | 动物: 无 | 物品: 无"
            latest_entity_overlay_text = to_ascii_overlay_text(latest_entity_console_text)
            reset_tracking_state()
            continue

        pressed = chr(key) if key != 255 else ""
        if pressed in {"1", "2", "3", "4"}:
            current_mode = int(pressed)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
