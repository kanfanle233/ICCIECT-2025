"""
推理模块
包含模型加载、可视化绘制、结果格式化、图片/视频/摄像头推理
"""

import os
import tempfile
import time
import uuid

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO

from config import COCO_ZH, IMAGENET_ZH, TASK_INFO, PALETTE, DEVICE
from session import is_alive, new_session, try_run, release_run
from utils import cv2_puttext_zh, get_label


# ========== 模型缓存 ==========
# 使用字典缓存模型的原因：
# 1. 同一个任务会被反复点击，如果每次都重新加载模型会非常慢。
# 2. key 是任务名，value 是已经放到目标设备上的 YOLO 模型实例。
_models = {}


def get_model(task):
    """按需加载并缓存 YOLO 模型（自动使用检测到的计算设备）"""
    if task not in _models:
        print(f"[加载模型] {TASK_INFO[task]['model']} (device={DEVICE}) ...")
        _models[task] = YOLO(TASK_INFO[task]["model"]).to(DEVICE)
        print(f"[加载完成]")
    return _models[task]


# ========== 图像转换 ==========
def _to_pil_and_bgr(image):
    """将各种图像输入转换为 PIL 和 BGR 格式"""
    if isinstance(image, str):
        bgr = cv2.imread(image)
        if bgr is None:
            raise ValueError(f"无法读取图像: {image}")
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), bgr
    elif isinstance(image, Image.Image):
        return image, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("空图像数组")
        return Image.fromarray(image), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    raise TypeError(f"不支持的输入类型: {type(image)}")


# ========== 可视化绘制 ==========
def draw_detection(img, results, task):
    """根据任务类型在图像上绘制检测结果"""
    out = img.copy()
    if out is None or out.size == 0:
        return out
    h, w = out.shape[:2]
    scale = max(h, w) / 640
    thick = max(2, int(scale))
    found = False

    if task == "detection":
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            found = True
            for j in range(len(boxes)):
                try:
                    c = PALETTE[j % len(PALETTE)]
                    x1, y1, x2, y2 = map(int, boxes.xyxy[j].cpu().numpy())
                    cv2.rectangle(out, (x1, y1), (x2, y2), c, thick)
                    lb = f"{get_label(r, boxes.cls[j], COCO_ZH)} {float(boxes.conf[j]):.2f}"
                    out = cv2_puttext_zh(
                        out,
                        lb,
                        (x1, max(y1 - 28, 0)),
                        fontsize=max(14, int(16 * scale)),
                    )
                except Exception:
                    continue

    elif task == "segmentation":
        for r in results:
            masks = getattr(r, "masks", None)
            if masks is None or len(masks) == 0:
                continue
            found = True
            boxes = getattr(r, "boxes", None)
            for j in range(len(masks.data)):
                try:
                    mask = cv2.resize(masks.data[j].cpu().numpy(), (w, h))
                    mb = (mask > 0.5).astype(np.uint8) * 255
                    colored = np.zeros_like(out)
                    colored[mb > 0] = PALETTE[j % len(PALETTE)]
                    out = cv2.addWeighted(out, 1.0, colored, 0.4, 0)
                    if boxes is not None and j < len(boxes):
                        lb = f"{get_label(r, boxes.cls[j], COCO_ZH)} {float(boxes.conf[j]):.2f}"
                        cnt, _ = cv2.findContours(
                            mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        if cnt:
                            M = cv2.moments(cnt[0])
                            if M["m00"] > 0:
                                cx, cy = (
                                    int(M["m10"] / M["m00"]),
                                    int(M["m01"] / M["m00"]),
                                )
                                out = cv2_puttext_zh(
                                    out,
                                    lb,
                                    (cx - 30, cy),
                                    fontsize=max(12, int(14 * scale)),
                                )
                except Exception:
                    continue

    elif task == "classification":
        if results and len(results) > 0:
            probs = getattr(results[0], "probs", None)
            if probs is not None and hasattr(probs, "top5") and len(probs.top5) > 0:
                found = True
                yo = 30
                for idx, conf in zip(probs.top5, probs.top5conf):
                    lb = f"{get_label(results[0], idx, IMAGENET_ZH)}: {float(conf):.1%}"
                    out = cv2_puttext_zh(
                        out, lb, (20, yo), fontsize=max(16, int(20 * scale))
                    )
                    yo += int(30 * scale)

    elif task == "pose":
        SKELETON = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ]
        for r in results:
            kp = getattr(r, "keypoints", None)
            if kp is None or kp.data is None or len(kp.data) == 0:
                continue
            found = True
            arr = kp.data.cpu().numpy()
            for person in arr:
                for sk in SKELETON:
                    p1, p2 = sk[0] - 1, sk[1] - 1
                    if p1 < len(person) and p2 < len(person):
                        x1, y1, c1 = (
                            float(person[p1][0]),
                            float(person[p1][1]),
                            float(person[p1][2]),
                        )
                        x2, y2, c2 = (
                            float(person[p2][0]),
                            float(person[p2][1]),
                            float(person[p2][2]),
                        )
                        if c1 > 0.3 and c2 > 0.3:
                            cv2.line(
                                out,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (0, 200, 255),
                                thick,
                            )
                for pt in person:
                    x, y, c = float(pt[0]), float(pt[1]), float(pt[2])
                    if c > 0.3:
                        cv2.circle(
                            out,
                            (int(x), int(y)),
                            max(3, int(4 * scale)),
                            (0, 200, 255),
                            -1,
                        )

    elif task == "obb":
        for r in results:
            obb = getattr(r, "obb", None)
            if obb is None or len(obb) == 0:
                continue
            found = True
            for j in range(len(obb)):
                try:
                    c = PALETTE[j % len(PALETTE)]
                    pts = obb.xyxyxyxy[j].cpu().numpy().astype(int).reshape((-1, 1, 2))
                    cv2.polylines(out, [pts], True, c, thick)
                    bw, bh = float(obb.xywhr[j][2]), float(obb.xywhr[j][3])
                    if bw > 60 and bh > 30:
                        lb = f"{get_label(r, obb.cls[j], COCO_ZH)} {float(obb.conf[j]):.2f}"
                        cx, cy = int(obb.xywhr[j][0]), int(obb.xywhr[j][1])
                        fs = max(10, min(14, int(min(bw, bh) / 4)))
                        out = cv2_puttext_zh(
                            out, lb, (cx - 30, max(cy - 10, fs)), fontsize=fs
                        )
                except Exception:
                    continue

    if not found:
        out = cv2_puttext_zh(out, "未检测到目标", (w // 2 - 80, h // 2), fontsize=24)
    return out


# ========== 结果格式化 ==========
def _collect_items(results, task):
    """收集检测项的元信息。

    这里统一返回“由字典组成的列表”，是因为不同任务的输出结构不同：
    - 检测/分割更关心标签、置信度、边界框。
    - 姿态更关心可见关键点个数。
    - 分类更关心 top-k 概率。
    用字典可以让初学者直接看到“字段名 -> 含义”的对应关系。
    """
    items = []
    for r in results:
        if task in ("detection", "segmentation"):
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for j in range(len(boxes)):
                try:
                    items.append(
                        {
                            "label": get_label(r, boxes.cls[j], COCO_ZH),
                            "conf": float(boxes.conf[j]),
                            "box": boxes.xyxy[j].cpu().numpy().astype(int)
                            if task == "detection"
                            else None,
                        }
                    )
                except Exception:
                    continue
        elif task == "classification":
            probs = getattr(r, "probs", None)
            if probs is not None and hasattr(probs, "top5"):
                for idx, conf in zip(probs.top5, probs.top5conf):
                    items.append(
                        {"label": get_label(r, idx, IMAGENET_ZH), "conf": float(conf)}
                    )
        elif task == "pose":
            kp = getattr(r, "keypoints", None)
            if kp is not None and kp.data is not None:
                for person in kp.data.cpu().numpy():
                    vis = sum(1 for p in person if float(p[2]) > 0.3)
                    items.append({"visible": vis, "total": len(person)})
        elif task == "obb":
            obb = getattr(r, "obb", None)
            if obb is None:
                continue
            for j in range(len(obb)):
                try:
                    items.append(
                        {
                            "label": get_label(r, obb.cls[j], COCO_ZH),
                            "conf": float(obb.conf[j]),
                            "angle": float(obb.xywhr[j][4]),
                        }
                    )
                except Exception:
                    continue
    return items


def format_results_text(results, task):
    """将推理结果格式化为文本"""
    items = _collect_items(results, task)
    if not items:
        return "未检测到结果"
    lines = []
    if task == "detection":
        lines.append(f"检测到 {len(items)} 个目标")
        for i, it in enumerate(items, 1):
            b = it["box"]
            lines.append(
                f"  {i}. {it['label']} -- 置信度 {it['conf']:.2%} -- 框 [{b[0]},{b[1]}] -> [{b[2]},{b[3]}]"
            )
    elif task == "segmentation":
        lines.append(f"分割到 {len(items)} 个区域")
        for i, it in enumerate(items, 1):
            lines.append(f"  {i}. {it['label']} -- 置信度 {it['conf']:.2%}")
    elif task == "classification":
        lines.append("Top-5 分类结果：")
        for it in items:
            lines.append(f"  {it['label']}: {it['conf']:.1%}")
    elif task == "pose":
        lines.append(f"检测到 {len(items)} 个人体")
        for i, it in enumerate(items, 1):
            lines.append(f"  人体 {i}: {it['visible']}/{it['total']} 个关键点可见")
    elif task == "obb":
        lines.append(f"检测到 {len(items)} 个定向目标")
        for i, it in enumerate(items, 1):
            lines.append(
                f"  {i}. {it['label']} -- 置信度 {it['conf']:.2%} -- 角度 {it['angle']:.1f}deg"
            )
    return "\n".join(lines)


# ========== 推理接口 ==========
def predict_image(image, task):
    """单张图片推理"""
    if image is None:
        return None, "请上传图像"
    try:
        model = get_model(task)
        pil, bgr = _to_pil_and_bgr(image)
        vis = draw_detection(bgr, model(pil, verbose=False), task)
        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), format_results_text(
            model(pil, verbose=False), task
        )
    except Exception as e:
        print(f"[predict_image 错误] {e}")
        import traceback

        traceback.print_exc()
        return None, f"推理出错: {str(e)}"


def predict_video_stream(path, task):
    """视频文件流式推理"""
    if path is None:
        yield None, "请上传视频"
        return
    if not try_run():
        yield None, "⚠️ 已有推理在运行"
        return
    sid = new_session()
    cap, writer = None, None
    try:
        model = get_model(task)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            yield None, "无法打开视频"
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w, h = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        out_path = str(
            Path(tempfile.gettempdir()) / f"out_{task}_{uuid.uuid4().hex[:8]}.mp4"
        )
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
        idx, last_txt, t0 = 0, "", time.time()
        stopped = False
        while is_alive(sid):
            ret, frame = cap.read()
            if not ret:
                break
            if not is_alive(sid):
                stopped = True
                break
            res = model(frame, verbose=False)
            if not is_alive(sid):
                stopped = True
                break
            vis = draw_detection(frame, res, task)
            if writer and writer.isOpened():
                writer.write(vis)
            if idx % max(int(fps), 1) == 0:
                last_txt = format_results_text(res, task)
            yield (
                cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                f"▶ 第 {idx}/{total} 帧 | {time.time() - t0:.1f}s\n\n{last_txt}",
            )
            idx += 1
        if stopped:
            yield None, "⏹ 已停止"
        else:
            yield (
                None,
                (
                    f"✅ 完成 {idx} 帧\n\n{last_txt}"
                    + (
                        f"\n📁 {out_path}"
                        if out_path and os.path.exists(out_path)
                        else ""
                    )
                ),
            )
    except Exception as e:
        print(f"[video_stream 错误] {e}")
        import traceback

        traceback.print_exc()
        yield None, f"视频处理出错: {str(e)}"
    finally:
        if cap and cap.isOpened():
            cap.release()
        if writer and writer.isOpened():
            writer.release()
        release_run()


def predict_webcam_stream(task):
    """摄像头实时流式推理"""
    if not try_run():
        yield None, "⚠️ 已有推理在运行"
        return
    sid = new_session()
    cap = None
    try:
        model = get_model(task)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            yield None, "❌ 无法打开摄像头"
            return
        for _ in range(3):
            cap.read()
        cnt, last, t0 = 0, "", time.time()
        yield None, "📹 正在启动实时推理…"
        while is_alive(sid):
            ret, frame = cap.read()
            if not ret:
                yield None, "❌ 摄像头断开"
                break
            if not is_alive(sid):
                break
            res = model(frame, verbose=False)
            if not is_alive(sid):
                break
            vis = draw_detection(frame, res, task)
            if cnt % 10 == 0:
                last = format_results_text(res, task)
            cnt += 1
            fps = cnt / (time.time() - t0) if time.time() - t0 > 0 else 0
            yield (
                cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                f"📹 {cnt} 帧 | {fps:.1f} FPS\n\n{last}",
            )
        yield None, "⏹ 已停止"
    except Exception as e:
        print(f"[webcam_stream 错误] {e}")
        import traceback

        traceback.print_exc()
        yield None, f"摄像头出错: {str(e)}"
    finally:
        if cap and cap.isOpened():
            cap.release()
        release_run()
