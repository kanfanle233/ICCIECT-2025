"""
人脸表情识别教学脚本。

这个项目和 YOLO 示例最大的不同是：DeepFace 底层依赖 TensorFlow，
它通常由框架自己决定后端，而不是像 Ultralytics 那样每次调用都显式传 `device=`。
因此本文件重点讲清三件事：
1. 为什么要把 `.deepface` 缓存重定向到项目目录。
2. TensorFlow 当前实际选择了什么物理设备。
3. 一帧图像是如何从摄像头进入，再被识别并绘制中文标签的。
"""

from pathlib import Path
import os

CURRENT_DIR = Path(__file__).resolve().parent
DEEPFACE_HOME = CURRENT_DIR / ".deepface"
DEEPFACE_WEIGHTS_DIR = DEEPFACE_HOME / "weights"
FONT_PATH = CURRENT_DIR / "msyh.ttc"

# 先重定向 HOME，再导入 DeepFace。
# 原因：DeepFace 初始化时会尝试在 HOME 下创建 `.deepface` 目录；
# 如果放任它写到用户主目录，课堂环境里很容易遇到权限或路径不一致问题。
os.environ["HOME"] = str(CURRENT_DIR)
os.environ["USERPROFILE"] = str(CURRENT_DIR)
DEEPFACE_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from deepface import DeepFace  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

EMOTION_MAP = {
    "angry": "愤怒",
    "disgust": "厌恶",
    "fear": "恐惧",
    "happy": "开心",
    "sad": "悲伤",
    "surprise": "惊讶",
    "neutral": "平静",
}


def detect_tensorflow_backend():
    """读取 TensorFlow 当前实际暴露出来的物理设备列表。"""
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - 依赖是否可用由本地环境决定
        return {
            "display": "TensorFlow backend unavailable",
            "reason": f"无法导入 TensorFlow，因此无法进一步判断后端：{exc}",
        }

    devices = tf.config.list_physical_devices()
    device_types = [device.device_type for device in devices]

    if "GPU" in device_types:
        return {
            "display": "TensorFlow GPU backend",
            "reason": "TensorFlow 识别到了 GPU；DeepFace 会由框架自己把计算图放到这个后端。",
        }

    return {
        "display": "TensorFlow CPU backend",
        "reason": "当前 TensorFlow 只暴露了 CPU 设备，因此 DeepFace 会自动回退到 CPU 执行。",
    }


def load_font():
    """加载中文字体，避免每一帧都重复创建字体对象。"""
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"找不到字体文件: {FONT_PATH}")
    return ImageFont.truetype(str(FONT_PATH), 30)


def draw_chinese_label(frame, x, y, label_cn, font):
    """用 PIL 绘制中文，绕开 OpenCV 对中文渲染不稳定的问题。"""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y - 45), f"识别结果: {label_cn}", font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def start_app():
    """启动人脸表情识别应用：打开摄像头，实时检测人脸并显示中文情绪标签。"""
    font = load_font()
    backend_info = detect_tensorflow_backend()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头，请检查摄像头权限或占用情况。")

    print("系统启动成功！正在读取本地模型... 按 'q' 键退出")
    print(f"推理后端: {backend_info['display']}")
    print(f"后端说明: {backend_info['reason']}")
    print("说明: DeepFace/ TensorFlow 的后端由框架自动选择，这里只负责检查并报告实际结果。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取摄像头画面")
            break

        try:
            results = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )

            # DeepFace 有时返回单个字典，有时返回字典列表；
            # 这里统一包装成列表，后面的循环就不用分两套分支。
            if isinstance(results, dict):
                results = [results]

            for res in results:
                region = res.get("region", {})
                x = region.get("x", 0)
                y = region.get("y", 0)
                w = region.get("w", 0)
                h = region.get("h", 0)

                emotion_en = res.get("dominant_emotion", "")
                label_cn = EMOTION_MAP.get(emotion_en, "未知")

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                frame = draw_chinese_label(frame, x, y, label_cn, font)

        except Exception as exc:
            # 识别失败时继续展示原视频流，避免摄像头窗口直接崩溃退出。
            print(f"本帧识别跳过: {exc}")

        cv2.imshow("Face Emotion Recognition System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_app()
