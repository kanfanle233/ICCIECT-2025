"""
工具函数模块
包含样本列表、字体加载、中文绘图、标签获取等功能
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import IMAGES_DIR, VIDEOS_DIR, PALETTE


def list_samples():
    """列出样本目录中的图片和视频文件。

    返回值是两个列表 `(imgs, vids)`。
    这样界面层可以直接把列表交给下拉框，而不用再次解析目录。
    """
    img_exts = (
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.gif",
        "*.webp",
        "*.tiff",
        "*.tif",
    )
    vid_exts = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv", "*.flv", "*.webm")
    imgs = sorted(p.name for ext in img_exts for p in IMAGES_DIR.glob(ext))
    vids = sorted(p.name for ext in vid_exts for p in VIDEOS_DIR.glob(ext))
    return imgs, vids


def _get_font(size):
    """获取中文字体"""
    paths = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    if sys.platform == "win32":
        paths += [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


# 字体文件的加载成本较高，所以这里用字典做缓存：
# key 是字号，value 是已经加载好的字体对象。
_font_cache = {}


def cv2_puttext_zh(img, text, pos, color=(255, 255, 255), fontsize=18):
    """在 OpenCV 图像上绘制中文文字"""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _font_cache.setdefault(fontsize, _get_font(fontsize))
    x, y = pos
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([x, y, x + tw + 8, y + th + 6], fill=(30, 64, 175, 220))
    draw.text((x + 4, y + 2), text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def get_label(r, cls_id, zh_map):
    """获取中文/英文标签"""
    name = r.names.get(int(cls_id), f"类别{int(cls_id)}")
    return zh_map.get(name, name)


def export_to_coreml(task, output_dir=None):
    """将 YOLO 模型导出为 Core ML 格式 (.mlpackage)，利用 Apple Neural Engine 加速

    导出后的模型可通过 YOLO(coreml_model_path) 直接加载使用，推理速度可提升 3-8 倍。
    需要: pip install coremltools
    """
    try:
        import coremltools  # noqa: F401
    except ImportError:
        raise ImportError(
            "导出 Core ML 需要 coremltools 库。请执行: pip install coremltools"
        )
    from ultralytics import YOLO

    from config import TASK_INFO

    if task not in TASK_INFO:
        raise ValueError(f"未知任务: {task}，可选: {list(TASK_INFO.keys())}")
    model_file = TASK_INFO[task]["model"]
    output_dir = Path(output_dir or Path(__file__).parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Core ML 导出] 正在将 {model_file} 导出为 Core ML ...")
    model = YOLO(model_file)
    out_path = model.export(format="coreml", nms=True, imgsz=640)
    print(f"[Core ML 导出] 完成: {out_path}")
    return str(out_path)
