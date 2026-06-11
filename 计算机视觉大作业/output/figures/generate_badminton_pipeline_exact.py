from __future__ import annotations

import base64
import csv
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent
ROOT_DIR = OUT_DIR.parents[1]
ASSET_DIR = OUT_DIR / "exact_assets"

DRAWIO = OUT_DIR / "badminton_pipeline_exact.drawio"
SVG = OUT_DIR / "badminton_pipeline_exact.svg"
PNG = OUT_DIR / "badminton_pipeline_exact.png"

W, H = 1732, 964
SCALE = 2
PW, PH = W * SCALE, H * SCALE

INK = "#111111"
MUTED = "#333333"
LIGHT_STROKE = "#bfc7d5"
DASH = "#111111"
BLUE_FILL = "#eaf2ff"
GREEN_FILL = "#ecf8e6"
YELLOW_FILL = "#fff5d9"
PINK_FILL = "#fde7ea"
PURPLE_FILL = "#f1effa"
WHITE = "#ffffff"


def sp(v: float) -> int:
    return int(round(v * SCALE))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, sp(size))
        except OSError:
            pass
    return ImageFont.load_default()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def project_data() -> dict:
    manifest = load_json(ROOT_DIR / "frontend/public/data/manifest.json")
    video_id = manifest.get("default_video") or "pro_match19_1_01_01"
    video_dir = ROOT_DIR / "frontend/public/data/videos" / video_id
    analysis = load_json(video_dir / "analysis.json")
    quality = load_json(video_dir / "quality.json")
    all_scores = [
        v["analysis_meta"]["ball_quality_score"]
        for v in manifest.get("videos", [])
        if v.get("analysis_meta", {}).get("ball_quality_score") is not None
    ]
    mean_quality = sum(all_scores) / len(all_scores) if all_scores else analysis.get("analysis_meta", {}).get("ball_quality_score", 93.81)
    return {
        "manifest": manifest,
        "video_id": video_id,
        "video_dir": video_dir,
        "analysis": analysis,
        "quality": quality,
        "mean_quality": mean_quality,
        "ball_rows": read_rows(video_dir / "ball.csv"),
        "player_rows": read_rows(video_dir / "players.csv"),
        "motion_rows": read_rows(video_dir / "motion.csv"),
    }


def image_data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def pil_data_uri(img: Image.Image) -> str:
    from io import BytesIO

    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def quicklook_thumbnail(video_path: Path, name: str) -> Path | None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["qlmanage", "-t", "-s", "640", "-o", str(ASSET_DIR), str(video_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except Exception:
        return None
    generated = ASSET_DIR / f"{video_path.name}.png"
    if not generated.exists():
        return None
    target = ASSET_DIR / f"{name}.png"
    generated.replace(target)
    return target


def cover_image(path: Path, width: int, height: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    src_w, src_h = img.size
    scale = max(width / src_w, height / src_h)
    resized = img.resize((int(src_w * scale), int(src_h * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def prepare_assets(data: dict) -> dict[str, Image.Image]:
    docs = ROOT_DIR / "docs/images"
    video_dir = data["video_dir"]
    original = quicklook_thumbnail(video_dir / "original.mp4", "original") or docs / "01_input_frame.jpg"
    overlay = quicklook_thumbnail(video_dir / "overlay.mp4", "overlay") or docs / "03_tracknet_output.jpg"
    final = quicklook_thumbnail(video_dir / "final.mp4", "final") or docs / "05_panel_close.jpg"
    dashboard = docs / "frontend_dashboard.png"
    input_frame = docs / "01_input_frame.jpg"
    tracknet_frame = docs / "03_tracknet_output.jpg"
    return {
        "broadcast": cover_image(input_frame, 150, 88),
        "original": cover_image(original, 190, 112),
        "tracknet": cover_image(tracknet_frame, 205, 112),
        "overlay": cover_image(overlay, 205, 112),
        "final": cover_image(final, 205, 112),
        "dashboard": cover_image(dashboard, 260, 142),
    }


def draw_rounded(draw: ImageDraw.ImageDraw, xy, radius, fill, outline=INK, width=2, dash=False):
    xy2 = [sp(xy[0]), sp(xy[1]), sp(xy[2]), sp(xy[3])]
    if dash:
        draw_dashed_rect(draw, xy, outline, width)
    else:
        draw.rounded_rectangle(xy2, radius=sp(radius), fill=fill, outline=outline, width=sp(width))


def draw_dashed_rect(draw: ImageDraw.ImageDraw, xy, color=INK, width=2, dash=8, gap=6):
    x1, y1, x2, y2 = [sp(v) for v in xy]
    dash_s, gap_s = sp(dash), sp(gap)
    for x in range(x1, x2, dash_s + gap_s):
        draw.line([(x, y1), (min(x + dash_s, x2), y1)], fill=color, width=sp(width))
        draw.line([(x, y2), (min(x + dash_s, x2), y2)], fill=color, width=sp(width))
    for y in range(y1, y2, dash_s + gap_s):
        draw.line([(x1, y), (x1, min(y + dash_s, y2))], fill=color, width=sp(width))
        draw.line([(x2, y), (x2, min(y + dash_s, y2))], fill=color, width=sp(width))


def draw_text(draw: ImageDraw.ImageDraw, xy, text: str, size=14, bold=False, fill=INK, anchor=None, align="center"):
    draw.text((sp(xy[0]), sp(xy[1])), text, font=font(size, bold), fill=fill, anchor=anchor, align=align)


def draw_center(draw, box, text, size=13, bold=False, fill=INK):
    x1, y1, x2, y2 = box
    draw_text(draw, ((x1 + x2) / 2, (y1 + y2) / 2), text, size, bold, fill, anchor="mm")


def draw_arrow(draw: ImageDraw.ImageDraw, points, dashed=False, width=2):
    pts = [(sp(x), sp(y)) for x, y in points]
    if dashed:
        for p1, p2 in zip(pts[:-1], pts[1:]):
            draw_dashed_line(draw, p1, p2, width=width)
    else:
        draw.line(pts, fill=INK, width=sp(width), joint="curve")
    x1, y1 = pts[-2]
    x2, y2 = pts[-1]
    angle = math.atan2(y2 - y1, x2 - x1)
    size = sp(10)
    p1 = (x2, y2)
    p2 = (x2 - size * math.cos(angle - 0.45), y2 - size * math.sin(angle - 0.45))
    p3 = (x2 - size * math.cos(angle + 0.45), y2 - size * math.sin(angle + 0.45))
    draw.polygon([p1, p2, p3], fill=INK)


def draw_dashed_line(draw: ImageDraw.ImageDraw, p1, p2, width=2, dash=8, gap=6):
    x1, y1 = p1
    x2, y2 = p2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist == 0:
        return
    step = sp(dash + gap)
    dash_len = sp(dash)
    count = max(1, int(dist / step))
    for i in range(count + 1):
        start = i * step
        end = min(start + dash_len, dist)
        if start >= dist:
            break
        a, b = start / dist, end / dist
        draw.line([(x1 + (x2 - x1) * a, y1 + (y2 - y1) * a), (x1 + (x2 - x1) * b, y1 + (y2 - y1) * b)], fill=INK, width=sp(width))


def paste_img(canvas: Image.Image, img: Image.Image, xy):
    canvas.paste(img.resize((sp(img.width / SCALE) if img.width > 400 else sp(img.width), sp(img.height / SCALE) if img.height > 300 else sp(img.height))))


def place_image(canvas: Image.Image, img: Image.Image, box):
    x1, y1, x2, y2 = [sp(v) for v in box]
    resized = img.resize((x2 - x1, y2 - y1), Image.Resampling.LANCZOS)
    canvas.paste(resized, (x1, y1))


def draw_stage_title(draw, x, y, number, first, second=None, third=None):
    title_font = font(22, True)
    bbox = draw.textbbox((0, 0), first, font=title_font)
    title_w = (bbox[2] - bbox[0]) / SCALE
    circle_x = x - title_w / 2 - 18
    draw.ellipse([sp(circle_x - 12), sp(y - 12), sp(circle_x + 12), sp(y + 12)], fill=WHITE, outline=INK, width=sp(2))
    draw_text(draw, (circle_x, y), str(number), 15, True, anchor="mm")
    draw_text(draw, (x, y), first, 22, True, anchor="mm")
    if second:
        draw_text(draw, (x, y + 24), second, 16, True, anchor="mm")
    if third:
        draw_text(draw, (x, y + 44), third, 16, True, anchor="mm")


def draw_film_frame(draw, box):
    x1, y1, x2, y2 = box
    draw.rectangle([sp(x1), sp(y1), sp(x2), sp(y2)], outline=INK, width=sp(2))
    for y in range(int(y1 + 7), int(y2 - 5), 12):
        draw.rectangle([sp(x1 + 3), sp(y), sp(x1 + 12), sp(y + 6)], fill=WHITE, outline="#888888", width=sp(1))
        draw.rectangle([sp(x2 - 12), sp(y), sp(x2 - 3), sp(y + 6)], fill=WHITE, outline="#888888", width=sp(1))


def draw_court(draw, box, keypoints=True, top=False):
    x1, y1, x2, y2 = box
    draw.rectangle([sp(x1), sp(y1), sp(x2), sp(y2)], fill="#fbfbfb", outline="#777777", width=sp(1))
    for x in [x1 + (x2 - x1) * 0.25, x1 + (x2 - x1) * 0.5, x1 + (x2 - x1) * 0.75]:
        draw.line([(sp(x), sp(y1)), (sp(x), sp(y2))], fill="#999999", width=sp(1))
    for y in [y1 + (y2 - y1) * 0.33, y1 + (y2 - y1) * 0.66]:
        draw.line([(sp(x1), sp(y)), (sp(x2), sp(y))], fill="#999999", width=sp(1))
    if keypoints:
        points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        colors = ["#d22", "#2d5be3", "#f4d000", "#31a83f"]
        for (x, y), c in zip(points, colors):
            draw.ellipse([sp(x - 4), sp(y - 4), sp(x + 4), sp(y + 4)], fill=c)
    if top:
        yellow = [(x1 + 35, y2 - 30), (x1 + 55, y2 - 42), (x1 + 70, y2 - 62), (x1 + 82, y2 - 88), (x1 + 93, y2 - 115)]
        blue = [(x2 - 35, y2 - 48), (x2 - 55, y2 - 58), (x2 - 78, y2 - 82), (x2 - 98, y2 - 105)]
        draw.line([(sp(x), sp(y)) for x, y in yellow], fill="#f4d000", width=sp(3))
        draw.line([(sp(x), sp(y)) for x, y in blue], fill="#3674ff", width=sp(3))
        for x, y in yellow:
            draw.ellipse([sp(x - 3), sp(y - 3), sp(x + 3), sp(y + 3)], fill="#f4d000")
        for x, y in blue:
            draw.ellipse([sp(x - 3), sp(y - 3), sp(x + 3), sp(y + 3)], fill="#3674ff")


def draw_trapezoid(draw, box, label, fill=BLUE_FILL):
    x1, y1, x2, y2 = box
    pts = [(sp(x1), sp(y1)), (sp(x2), sp(y1)), (sp(x2 - 12), sp(y2)), (sp(x1 + 12), sp(y2))]
    draw.polygon(pts, fill=fill, outline="#6c8dbf")
    draw_center(draw, box, label, 13, True)


def chart_points(rows, key_x="x_px", key_y="y_px", max_points=42):
    pts = []
    for row in rows:
        try:
            if row.get("visibility") in {"0", "0.0"}:
                continue
            x = float(row[key_x])
            y = float(row[key_y])
        except Exception:
            continue
        pts.append((x, y, row.get("source", "")))
    if not pts:
        return []
    step = max(1, len(pts) // max_points)
    return pts[::step][:max_points]


def draw_scatter(draw, box, rows, refined=False):
    x1, y1, x2, y2 = box
    draw.rectangle([sp(x1), sp(y1), sp(x2), sp(y2)], fill="#ffffff", outline="#777777", width=sp(1))
    for i in range(1, 5):
        xx = x1 + (x2 - x1) * i / 5
        yy = y1 + (y2 - y1) * i / 5
        draw.line([(sp(xx), sp(y1)), (sp(xx), sp(y2))], fill="#e1e1e1", width=sp(1))
        draw.line([(sp(x1), sp(yy)), (sp(x2), sp(yy))], fill="#e1e1e1", width=sp(1))
    pts = chart_points(rows, max_points=36)
    if not pts:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    mapped = []
    for px, py, src in pts:
        xx = x1 + 12 + (px - min_x) / max(max_x - min_x, 1) * (x2 - x1 - 24)
        yy = y1 + 10 + (py - min_y) / max(max_y - min_y, 1) * (y2 - y1 - 20)
        mapped.append((xx, yy, src))
    if len(mapped) > 1:
        draw.line([(sp(x), sp(y)) for x, y, _ in mapped], fill="#2e6df0" if not refined else "#419345", width=sp(1))
    for i, (x, y, src) in enumerate(mapped):
        c = "#d33" if (not refined and i % 7 == 0) else ("#419345" if refined else "#2e6df0")
        draw.ellipse([sp(x - 3), sp(y - 3), sp(x + 3), sp(y + 3)], fill=c)


def draw_black_detections(draw, box, rows):
    x1, y1, x2, y2 = box
    draw.rectangle([sp(x1), sp(y1), sp(x2), sp(y2)], fill="#020202", outline=INK, width=sp(1))
    pts = chart_points(rows, max_points=26)
    if not pts:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    for px, py, _ in pts:
        xx = x1 + 10 + (px - min(xs)) / max(max(xs) - min(xs), 1) * (x2 - x1 - 20)
        yy = y1 + 10 + (py - min(ys)) / max(max(ys) - min(ys), 1) * (y2 - y1 - 20)
        draw.ellipse([sp(xx - 2), sp(yy - 2), sp(xx + 2), sp(yy + 2)], fill="#ffffff")


def draw_gauge(draw, center, radius, score, text=None):
    cx, cy = center
    bbox = [sp(cx - radius), sp(cy - radius), sp(cx + radius), sp(cy + radius)]
    for start, end, color in [(180, 225, "#e74c3c"), (225, 285, "#ffc928"), (285, 360, "#7bc96f")]:
        draw.arc(bbox, start, end, fill=color, width=sp(12))
    draw.arc(bbox, 180, 360, fill="#aaaaaa", width=sp(1))
    draw_text(draw, (cx, cy + 2), f"{score:.2f}", 20, True, anchor="mm")
    draw_text(draw, (cx + 34, cy + 24), "/100", 14, True, anchor="mm")
    if text:
        draw_text(draw, (cx, cy + 52), text, 14, False, "#1f7a2f", anchor="mm")


def draw_table(draw, box, headers, rows, title=None, plot=False):
    x1, y1, x2, y2 = box
    draw_rounded(draw, box, 5, "#fafbff", "#b5bfd1", 1)
    if title:
        draw_text(draw, ((x1 + x2) / 2, y1 + 18), title, 13, True, anchor="mm")
    tx1, ty1 = x1 + 12, y1 + 38
    col_w = (x2 - x1 - 24) / len(headers)
    row_h = 24
    for i, h in enumerate(headers):
        draw.rectangle([sp(tx1 + i * col_w), sp(ty1), sp(tx1 + (i + 1) * col_w), sp(ty1 + row_h)], fill="#f4f4f4", outline="#c7c7c7", width=sp(1))
        draw_center(draw, (tx1 + i * col_w, ty1, tx1 + (i + 1) * col_w, ty1 + row_h), h, 8, False)
    for r, vals in enumerate(rows):
        for i, val in enumerate(vals):
            y = ty1 + (r + 1) * row_h
            draw.rectangle([sp(tx1 + i * col_w), sp(y), sp(tx1 + (i + 1) * col_w), sp(y + row_h)], fill=WHITE, outline="#d3d3d3", width=sp(1))
            draw_center(draw, (tx1 + i * col_w, y, tx1 + (i + 1) * col_w, y + row_h), str(val), 8, False)
    if plot:
        px1, py1, px2, py2 = x2 - 72, y1 + 47, x2 - 16, y2 - 18
        draw.rectangle([sp(px1), sp(py1), sp(px2), sp(py2)], fill="#ffffff", outline="#d0d0d0", width=sp(1))
        pts = [(px1 + 8, py2 - 8), (px1 + 18, py2 - 30), (px1 + 30, py2 - 46), (px1 + 44, py2 - 64), (px1 + 55, py2 - 80)]
        draw.line([(sp(x), sp(y)) for x, y in pts], fill="#246bff", width=sp(2))
        for x, y in pts:
            draw.ellipse([sp(x - 2), sp(y - 2), sp(x + 2), sp(y + 2)], fill="#246bff")
        for x, y in pts[:2]:
            draw.ellipse([sp(x - 2), sp(y - 2), sp(x + 2), sp(y + 2)], fill="#d33")


def draw_file_icon(draw, box, label, color, kind):
    x1, y1, x2, y2 = box
    draw_rounded(draw, box, 8, "#ffffff", color, 3)
    if kind == "PLAY":
        cx, cy = (x1 + x2) / 2, y1 + 34
        pts = [(sp(cx - 10), sp(cy - 14)), (sp(cx - 10), sp(cy + 14)), (sp(cx + 14), sp(cy))]
        draw.polygon(pts, fill=color)
    else:
        draw_text(draw, ((x1 + x2) / 2, y1 + 36), kind, 18, True, color, anchor="mm")
    draw_text(draw, ((x1 + x2) / 2, y2 + 18), label, 11, False, INK, anchor="mm")


def draw_person_stick(draw, cx, cy, color):
    draw.ellipse([sp(cx - 5), sp(cy - 20), sp(cx + 5), sp(cy - 10)], outline=color, width=sp(3))
    draw.line([(sp(cx), sp(cy - 10)), (sp(cx), sp(cy + 10))], fill=color, width=sp(3))
    draw.line([(sp(cx - 16), sp(cy)), (sp(cx + 16), sp(cy - 2))], fill=color, width=sp(3))
    draw.line([(sp(cx), sp(cy + 10)), (sp(cx - 14), sp(cy + 28))], fill=color, width=sp(3))
    draw.line([(sp(cx), sp(cy + 10)), (sp(cx + 15), sp(cy + 28))], fill=color, width=sp(3))


def make_png():
    data = project_data()
    assets = prepare_assets(data)
    img = Image.new("RGB", (PW, PH), WHITE)
    draw = ImageDraw.Draw(img)
    rows = data["ball_rows"]
    analysis_meta = data["analysis"].get("analysis_meta", {})
    q_score = data["mean_quality"]
    dash_score = data["quality"].get("ball_quality_score", analysis_meta.get("ball_quality_score", q_score))

    # Titles.
    draw_stage_title(draw, 110, 27, 1, "Input")
    draw_stage_title(draw, 395, 26, 2, "Shuttlecock Detection", "(TrackNet)")
    draw_stage_title(draw, 780, 26, 3, "Trajectory Refinement", "& Quality Gating")
    draw_stage_title(draw, 1238, 26, 4, "Player Analysis", "(YOLOv8-Pose + ByteTrack)")
    draw_stage_title(draw, 1585, 26, 5, "Rendering", "& Export")

    # Stage dashed panels.
    panels = [(15, 82, 225, 625), (265, 82, 510, 625), (550, 82, 955, 625), (970, 82, 1405, 625), (1435, 82, 1715, 570)]
    for p in panels:
        draw_dashed_rect(draw, p, DASH, 2)

    # Input stage.
    draw_text(draw, (112, 110), "Broadcast Video", 13, True, anchor="mm")
    draw_film_frame(draw, (35, 130, 205, 225))
    place_image(img, assets["broadcast"], (50, 132, 192, 220))
    draw_text(draw, (112, 248), "...", 14, True, anchor="mm")
    draw_text(draw, (112, 286), "Court Keypoints", 13, True, anchor="mm")
    draw_court(draw, (55, 310, 190, 430), True)
    draw_text(draw, (112, 485), "Models", 14, True, anchor="mm")
    for i, label in enumerate(["TrackNet_best.pt", "yolov8s-pose.pt", "ByteTrack"]):
        y = 505 + i * 34
        draw.rectangle([sp(30), sp(y), sp(205), sp(y + 28)], fill="#e8f1ff", outline="#9fb4d0", width=sp(1))
        draw_center(draw, (30, y, 205, y + 28), label, 12)

    # Shuttle stage.
    draw_text(draw, (386, 112), "Input Clip (T frames)", 13, True, anchor="mm")
    for i in range(6):
        place_image(img, assets["original"], (286 + i * 11, 135 + i * 6, 458 + i * 11, 235 + i * 6))
    draw_arrow(draw, [(386, 260), (386, 282)], width=1.5)
    draw_trapezoid(draw, (302, 276, 478, 315), "TrackNet")
    draw_arrow(draw, [(386, 318), (386, 337)], width=1.5)
    draw_text(draw, (386, 349), "Heatmap (Center)", 13, True, anchor="mm")
    heat = Image.new("RGB", (205, 112), "#07134a")
    hdraw = ImageDraw.Draw(heat)
    hdraw.rectangle([0, 0, 205, 112], fill="#07134a")
    for r, c in [(32, "#001f8c"), (22, "#125dff"), (14, "#12d6ff"), (8, "#f6e100"), (4, "#ff3b20")]:
        hdraw.ellipse([102 - r, 62 - r, 102 + r, 62 + r], fill=c)
    place_image(img, heat, (284, 365, 489, 477))
    draw_arrow(draw, [(386, 477), (386, 495)], width=1.5)
    draw_text(draw, (386, 493), "Raw Detections", 13, True, anchor="mm")
    draw_black_detections(draw, (285, 510, 490, 600), rows)
    draw_text(draw, (386, 612), "...", 14, True, anchor="mm")

    # Trajectory refinement.
    draw_text(draw, (660, 110), "Raw Trajectory", 13, True, anchor="mm")
    draw_scatter(draw, (575, 132, 770, 222), rows, refined=False)
    draw_text(draw, (572, 175), "y", 12, False, anchor="mm")
    draw_text(draw, (766, 231), "x", 10, False, anchor="mm")
    draw_arrow(draw, [(675, 226), (675, 250)], width=1.5)
    draw_rounded(draw, (562, 250, 768, 465), 7, GREEN_FILL, "#8fb779", 1.5)
    draw_text(draw, (665, 270), "Refinement (Per Frame)", 13, True, anchor="mm")
    for i, label in enumerate(["ROI Filtering (Court Mask)", "Static Lock Removal", "Jump Rejection (Speed Th.)", "Kalman Smoothing", "Short Gap Interpolation"]):
        y = 288 + i * 36
        draw.rectangle([sp(568), sp(y), sp(762), sp(y + 27)], fill="#f7fff1", outline="#92b86d", width=sp(1))
        draw_center(draw, (568, y, 762, y + 27), label, 10)
    draw_arrow(draw, [(665, 465), (665, 490)], width=1.5)
    draw_text(draw, (665, 497), "Refined Trajectory", 13, True, anchor="mm")
    draw_scatter(draw, (575, 515, 770, 602), rows, refined=True)
    draw_text(draw, (572, 555), "y", 12, False, anchor="mm")
    draw_text(draw, (766, 610), "x", 10, False, anchor="mm")

    draw_arrow(draw, [(770, 195), (807, 195)], width=1.5)
    draw_rounded(draw, (808, 165, 940, 360), 7, "#f4ecff", "#9b79c9", 1.5)
    draw_text(draw, (874, 187), "Quality Scoring", 13, True, anchor="mm")
    for i, label in enumerate(["Visibility Ratio", "Max Gap (frames)", "Interpolation Ratio", "False Positive Rejection"]):
        y = 210 + i * 38
        draw.rectangle([sp(815), sp(y), sp(933), sp(y + 29)], fill="#efe4ff", outline="#a88acb", width=sp(1))
        draw_center(draw, (815, y, 933, y + 29), label, 10)
    draw_arrow(draw, [(874, 360), (874, 395)], width=1.5)
    draw_rounded(draw, (808, 405, 932, 525), 7, "#f7f2ff", "#9b79c9", 1.5)
    draw_text(draw, (870, 428), "Quality Score", 13, True, anchor="mm")
    draw_gauge(draw, (870, 485), 40, q_score)
    draw_rounded(draw, (808, 535, 932, 605), 7, "#f7f2ff", "#9b79c9", 1.5)
    draw_text(draw, (870, 555), "Quality Level", 13, True, anchor="mm")
    draw.rectangle([sp(826), sp(576), sp(915), sp(600)], fill="#f0ffe8", outline="#b1d2a3", width=sp(1))
    draw_center(draw, (826, 576, 915, 600), "Green", 14, False, "#1f7a2f")

    # Player analysis.
    draw_text(draw, (1092, 110), "YOLOv8-Pose", 13, True, anchor="mm")
    place_image(img, assets["tracknet"], (982, 128, 1188, 240))
    draw_person_stick(draw, 1038, 196, "#f3d71a")
    draw_person_stick(draw, 1128, 180, "#2f75ff")
    draw_arrow(draw, [(1092, 240), (1092, 265)], width=1.5)
    draw_text(draw, (1092, 286), "Detections", 11, True, anchor="mm")
    for i, x in enumerate([1010, 1055, 1100]):
        draw.rectangle([sp(x), sp(300), sp(x + 28), sp(372)], fill="#111111", outline="#888888", width=sp(1))
        draw_person_stick(draw, x + 14, 334, "#f3d71a" if i < 2 else "#2f75ff")
    draw_text(draw, (1150, 335), "...", 13, True, anchor="mm")
    draw_arrow(draw, [(1092, 372), (1092, 405)], width=1.5)
    draw_trapezoid(draw, (1010, 405, 1175, 442), "ByteTrack")
    draw_arrow(draw, [(1092, 444), (1092, 475)], width=1.5)
    draw_text(draw, (1092, 463), "Player Tracks (ID)", 11, True, anchor="mm")
    draw_text(draw, (990, 500), "ID 1", 11, True, anchor="mm")
    draw_text(draw, (990, 520), "(near)", 10, False, anchor="mm")
    draw_text(draw, (990, 562), "ID 2", 11, True, anchor="mm")
    draw_text(draw, (990, 582), "(far)", 10, False, anchor="mm")
    for row_i, color in enumerate(["#f3d71a", "#2f75ff"]):
        y = 492 + row_i * 62
        for i, x in enumerate([1022, 1070, 1118, 1172]):
            draw.rectangle([sp(x), sp(y), sp(x + 25), sp(y + 43)], fill="#111111", outline="#777777", width=sp(1))
            draw_person_stick(draw, x + 12, y + 20, color)
            if i < 3:
                draw_arrow(draw, [(x + 28, y + 22), (x + 43, y + 22)], dashed=i == 2, width=1)

    draw_arrow(draw, [(1189, 195), (1230, 195)], width=1.5)
    draw_text(draw, (1302, 110), "Homography", 13, True, anchor="mm")
    draw_text(draw, (1302, 128), "(Court Mapping)", 12, True, anchor="mm")
    draw_rounded(draw, (1238, 145, 1378, 270), 4, "#fbfbff", "#8581b4", 1.2)
    draw_court(draw, (1260, 165, 1358, 238), True)
    draw_text(draw, (1300, 253), "...", 10, True, anchor="mm")
    draw_arrow(draw, [(1302, 270), (1302, 310)], dashed=True, width=1.5)
    draw_text(draw, (1302, 298), "H", 16, True, anchor="mm")
    draw_text(draw, (1302, 330), "Top-down View", 12, True, anchor="mm")
    draw_court(draw, (1235, 342, 1388, 450), False, top=True)
    draw_arrow(draw, [(1302, 452), (1302, 485)], width=1.5)
    draw_rounded(draw, (1225, 490, 1392, 625), 7, "#eaf4ff", "#7c9cc3", 1.5)
    draw_text(draw, (1308, 510), "Motion Statistics", 12, True, anchor="mm")
    for i, label in enumerate(["Speed (m/s)", "Distance (m)", "Max Speed (m/s)", "..."]):
        y = 525 + i * 25
        draw.rectangle([sp(1238), sp(y), sp(1378), sp(y + 21)], fill="#f8fbff", outline="#89a4c3", width=sp(1))
        draw_center(draw, (1238, y, 1378, y + 21), label, 9)

    # Rendering and export.
    draw_text(draw, (1580, 110), "Overlay Video", 13, True, anchor="mm")
    place_image(img, assets["overlay"], (1472, 130, 1678, 242))
    draw_person_stick(draw, 1585, 187, "#2f75ff")
    draw_text(draw, (1580, 258), "...", 13, True, anchor="mm")
    draw_text(draw, (1580, 278), "Final Video", 13, True, anchor="mm")
    place_image(img, assets["final"], (1472, 295, 1678, 407))
    draw_person_stick(draw, 1585, 352, "#2f75ff")
    draw_arrow(draw, [(1580, 407), (1580, 438)], width=1.5)
    draw_rounded(draw, (1460, 448, 1690, 558), 7, "#f4f2ff", "#8581b4", 1.5)
    draw_text(draw, (1575, 470), "Data Export", 13, True, anchor="mm")
    draw_file_icon(draw, (1474, 486, 1528, 542), "CSV", "#44a060", "CSV")
    draw_file_icon(draw, (1548, 486, 1602, 542), "JSON", "#d0792d", "{...}")
    draw_file_icon(draw, (1622, 486, 1676, 542), "MP4", "#6d4aa0", "PLAY")
    draw_arrow(draw, [(1580, 558), (1580, 612)], width=1.5)

    # Cross-stage arrows.
    draw_arrow(draw, [(225, 355), (260, 355)], width=2)
    draw_arrow(draw, [(510, 355), (545, 355)], width=2)
    draw_arrow(draw, [(955, 640), (1298, 640), (1298, 625)], dashed=True, width=1)
    draw_arrow(draw, [(1405, 355), (1432, 355)], width=2)

    # Frontend dashboard.
    draw_rounded(draw, (1454, 620, 1720, 892), 6, PINK_FILL, "#e2a0aa", 1.5)
    draw_text(draw, (1587, 646), "Frontend Dashboard", 15, True, anchor="mm")
    draw_text(draw, (1587, 668), "(D3.js)", 13, True, anchor="mm")
    place_image(img, assets["dashboard"], (1465, 685, 1710, 880))
    draw_arrow(draw, [(1405, 770), (1448, 770)], width=2)

    # Bottom output files band.
    draw_dashed_rect(draw, (15, 665, 1405, 895), "#1c2b44", 2)
    ball_rows = data["ball_rows"]
    player_rows = [r for r in data["player_rows"] if r.get("source") == "detected"][:2]
    motion_rows = [r for r in data["motion_rows"] if r.get("speed_mps")][:2]
    draw_table(
        draw,
        (25, 682, 292, 862),
        ["frame", "x", "y", "visible"],
        [[0, "512.3", "256.7", 1], [1, "516.8", "261.2", 1], ["...", "", "", ""]],
        "Ball Trajectory (CSV)",
        True,
    )
    draw_table(
        draw,
        (305, 682, 582, 862),
        ["frame", "id", "x", "y", "foot_x", "foot_y"],
        [[0, 1, "120.3", "430.2", "118.6", "480.1"], [0, 2, "820.6", "410.8", "819.2", "459.3"], ["...", "", "", "", "", ""]],
        "Players (CSV)",
        False,
    )
    draw_table(
        draw,
        (595, 682, 858, 862),
        ["frame", "id", "speed", "vx", "vy"],
        [[0, 1, "0.00", "0.00", "0.00"], [1, 1, "1.32", "0.86", "1.00"], ["...", "", "", "", ""]],
        "Motion (CSV)",
        True,
    )
    draw_rounded(draw, (875, 682, 1138, 862), 5, "#fafbff", "#b5bfd1", 1)
    draw_text(draw, (1006, 700), "Stats (JSON)", 13, True, anchor="mm")
    stats_text = [
        '{',
        '  "video": "01.mp4",',
        '  "duration": 18.97,',
        '  "players": {',
        '    "1": {"distance": 78.34,',
        '          "max_speed": 8.21},',
        '    "2": {"distance": 71.42,',
        f'          "max_speed": {analysis_meta.get("max_speed_mps", 9.46):.2f}}}',
        "  }",
        "}",
    ]
    for i, line in enumerate(stats_text):
        draw_text(draw, (890, 720 + i * 14), line, 9, False, anchor="lm")
    # tiny bar chart
    bx, by = 1060, 765
    for i, (h, c) in enumerate([(42, "#67b7dc"), (32, "#f5a65b"), (55, "#725bd6")]):
        draw.rectangle([sp(bx + i * 24), sp(by + 55 - h), sp(bx + i * 24 + 12), sp(by + 55)], fill=c)
    draw_text(draw, (1087, 827), "Distance", 7, False, anchor="mm")
    draw_rounded(draw, (1152, 682, 1392, 862), 5, "#fafbff", "#b5bfd1", 1)
    draw_text(draw, (1272, 700), "Quality (JSON)", 13, True, anchor="mm")
    quality_lines = [
        "{",
        '  "video": "01.mp4",',
        f'  "quality_score": {dash_score:.2f},',
        '  "level": "Green",',
        '  "visibility": 0.72,',
        '  "max_gap": 4,',
        '  "interp_ratio": 0.08,',
        '  "fp_reject_ratio": 0.93',
        "}",
    ]
    for i, line in enumerate(quality_lines):
        draw_text(draw, (1168, 720 + i * 15), line, 9, False, anchor="lm")
    draw_gauge(draw, (1352, 778), 38, dash_score, "Green")

    # Legend.
    y = 925
    draw_arrow(draw, [(45, y), (88, y)], width=2)
    draw_text(draw, (100, y), "Data Flow", 11, False, anchor="lm")
    draw_arrow(draw, [(210, y), (255, y)], dashed=True, width=1.5)
    draw_text(draw, (263, y), "Module Flow", 11, False, anchor="lm")
    legend = [
        (385, BLUE_FILL, "Detection Module"),
        (585, GREEN_FILL, "Refinement Module"),
        (780, YELLOW_FILL, "Analysis Module"),
        (970, PINK_FILL, "Rendering/Export"),
        (1170, PURPLE_FILL, "Output Files"),
    ]
    for x, fill, label in legend:
        draw.rectangle([sp(x), sp(y - 10), sp(x + 38), sp(y + 10)], fill=fill, outline="#8a9ab5", width=sp(1))
        draw_text(draw, (x + 48, y), label, 11, False, anchor="lm")

    img = img.resize((W, H), Image.Resampling.LANCZOS)
    img.save(PNG)


def svg_image_tag(img: Image.Image, x, y, w, h):
    return f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="{pil_data_uri(img)}" preserveAspectRatio="xMidYMid slice"/>'


def make_svg():
    # Reuse the PNG as a visual backup, then keep the draw.io file as the editable source.
    # The PNG is embedded at full canvas size so SVG export remains a faithful picture.
    if not PNG.exists():
        make_png()
    data_uri = image_data_uri(PNG)
    SVG.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
                f'<image x="0" y="0" width="{W}" height="{H}" href="{data_uri}"/>',
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def mx_style(fill=WHITE, stroke=INK, rounded=False, dashed=False, extra=""):
    style = f"whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=1;"
    if rounded:
        style += "rounded=1;arcSize=8;"
    if dashed:
        style += "dashed=1;dashPattern=8 6;"
    return style + extra


def add_geo(cell, x, y, w, h):
    ET.SubElement(cell, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})


def make_drawio():
    data = project_data()
    assets = prepare_assets(data)
    analysis_meta = data["analysis"].get("analysis_meta", {})
    q_score = data["mean_quality"]
    dash_score = data["quality"].get("ball_quality_score", q_score)

    mx = ET.Element("mxfile", {"host": "app.diagrams.net", "agent": "Codex", "version": "24.7.17", "type": "device"})
    diagram = ET.SubElement(mx, "diagram", {"id": "badminton-pipeline-exact", "name": "Badminton Pipeline Exact"})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": str(W),
            "dy": str(H),
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": str(W),
            "pageHeight": str(H),
            "math": "0",
            "shadow": "0",
        },
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    def vertex(cid, value, style, x, y, w, h):
        cell = ET.SubElement(root, "mxCell", {"id": cid, "value": value, "style": style, "vertex": "1", "parent": "1"})
        add_geo(cell, x, y, w, h)
        return cell

    def text(cid, value, x, y, w, h, size=12, bold=False, align="center"):
        style = f"text;html=1;strokeColor=none;fillColor=none;align={align};verticalAlign=middle;fontFamily=Helvetica;fontSize={size};fontColor=#111111;"
        if bold:
            style += "fontStyle=1;"
        return vertex(cid, value, style, x, y, w, h)

    def rect(cid, x, y, w, h, fill=WHITE, stroke=INK, rounded=False, dashed=False):
        return vertex(cid, "", mx_style(fill, stroke, rounded, dashed), x, y, w, h)

    def image(cid, img: Image.Image, x, y, w, h):
        return vertex(cid, "", "shape=image;html=1;imageAspect=0;aspect=fixed;image=" + pil_data_uri(img) + ";", x, y, w, h)

    def edge(cid, source, target, dashed=False):
        style = "endArrow=block;html=1;rounded=0;strokeWidth=2;strokeColor=#111111;"
        if dashed:
            style += "dashed=1;dashPattern=8 6;"
        cell = ET.SubElement(root, "mxCell", {"id": cid, "value": "", "style": style, "edge": "1", "parent": "1", "source": source, "target": target})
        ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})

    def stage_header(cid, x, y, number, title, sub1=None, sub2=None):
        vertex(f"{cid}_circle", f"<b>{number}</b>", mx_style(WHITE, INK, False, False, "shape=ellipse;fontSize=14;fontStyle=1;fontFamily=Helvetica;align=center;verticalAlign=middle;"), x - 55, y - 13, 26, 26)
        text(cid, f"<b>{title}</b>", x - 24, y - 20, 230, 40, 22, True, "left")
        if sub1:
            text(f"{cid}_sub1", f"<b>{sub1}</b>", x - 24, y + 18, 230, 24, 16, True, "center")
        if sub2:
            text(f"{cid}_sub2", f"<b>{sub2}</b>", x - 24, y + 40, 230, 24, 16, True, "center")

    # Titles.
    stage_header("t1", 88, 27, 1, "Input")
    stage_header("t2", 335, 26, 2, "Shuttlecock Detection", "(TrackNet)")
    stage_header("t3", 720, 26, 3, "Trajectory Refinement", "&amp; Quality Gating")
    stage_header("t4", 1178, 26, 4, "Player Analysis", "(YOLOv8-Pose + ByteTrack)")
    stage_header("t5", 1545, 26, 5, "Rendering", "&amp; Export")

    for i, (x, y, w, h) in enumerate([(15, 82, 210, 543), (265, 82, 245, 543), (550, 82, 405, 543), (970, 82, 435, 543), (1435, 82, 280, 488)], 1):
        rect(f"panel_{i}", x, y, w, h, WHITE, INK, False, True)

    text("input_broadcast_label", "<b>Broadcast Video</b>", 40, 98, 160, 28, 13, True)
    rect("film_frame", 35, 130, 170, 95, WHITE, INK)
    image("broadcast_img", assets["broadcast"], 50, 132, 142, 88)
    text("input_ellipsis", "...", 88, 235, 50, 28, 14, True)
    text("court_label", "<b>Court Keypoints</b>", 40, 270, 160, 28, 13, True)
    rect("court_box", 55, 310, 135, 120, "#fbfbfb", "#777777")
    text("models_label", "<b>Models</b>", 40, 470, 160, 28, 14, True)
    for i, label in enumerate(["TrackNet_best.pt", "yolov8s-pose.pt", "ByteTrack"]):
        vertex(f"model_{i}", label, mx_style("#e8f1ff", "#9fb4d0", False, False, "fontSize=12;fontFamily=Helvetica;align=center;verticalAlign=middle;"), 30, 505 + i * 34, 175, 28)

    text("clip_label", "<b>Input Clip (T frames)</b>", 300, 98, 180, 28, 13, True)
    image("clip_stack", assets["original"], 290, 135, 190, 112)
    vertex("tracknet_trap", "<b>TrackNet</b>", "shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;fillColor=#eaf2ff;strokeColor=#6c8dbf;fontSize=13;fontFamily=Helvetica;align=center;verticalAlign=middle;", 302, 276, 176, 39)
    text("heatmap_label", "<b>Heatmap (Center)</b>", 305, 337, 180, 28, 13, True)
    image("heatmap_img", assets["tracknet"], 284, 365, 205, 112)
    text("raw_det_label", "<b>Raw Detections</b>", 305, 492, 180, 28, 13, True)
    rect("raw_det_box", 285, 510, 205, 90, "#050505", "#111111")

    text("raw_traj_label", "<b>Raw Trajectory</b>", 590, 98, 160, 28, 13, True)
    rect("raw_chart", 575, 132, 195, 90, WHITE, "#777777")
    rect("refine_box", 562, 250, 206, 215, GREEN_FILL, "#8fb779", True)
    text("refine_label", "<b>Refinement (Per Frame)</b>", 580, 257, 170, 30, 13, True)
    for i, label in enumerate(["ROI Filtering (Court Mask)", "Static Lock Removal", "Jump Rejection (Speed Th.)", "Kalman Smoothing", "Short Gap Interpolation"]):
        vertex(f"refine_step_{i}", label, mx_style("#f7fff1", "#92b86d", False, False, "fontSize=10;fontFamily=Helvetica;align=center;verticalAlign=middle;"), 568, 288 + i * 36, 194, 27)
    text("refined_label", "<b>Refined Trajectory</b>", 590, 482, 160, 28, 13, True)
    rect("refined_chart", 575, 515, 195, 87, WHITE, "#777777")
    rect("quality_scoring", 808, 165, 132, 195, "#f4ecff", "#9b79c9", True)
    text("quality_label", "<b>Quality Scoring</b>", 818, 174, 112, 30, 13, True)
    for i, label in enumerate(["Visibility Ratio", "Max Gap (frames)", "Interpolation Ratio", "False Positive Rejection"]):
        vertex(f"q_step_{i}", label, mx_style("#efe4ff", "#a88acb", False, False, "fontSize=10;fontFamily=Helvetica;align=center;verticalAlign=middle;"), 815, 210 + i * 38, 118, 29)
    rect("quality_score_box", 808, 405, 124, 120, "#f7f2ff", "#9b79c9", True)
    text("quality_score_label", "<b>Quality Score</b>", 820, 414, 100, 26, 13, True)
    text("quality_score_value", f"<b>{q_score:.2f}</b><br>/100", 820, 454, 100, 60, 20, True)
    rect("quality_level_box", 808, 535, 124, 70, "#f7f2ff", "#9b79c9", True)
    text("quality_level_label", "<b>Quality Level</b>", 820, 542, 100, 28, 13, True)
    vertex("green_label", "Green", mx_style("#f0ffe8", "#b1d2a3", False, False, "fontSize=14;fontFamily=Helvetica;fontColor=#1f7a2f;align=center;verticalAlign=middle;"), 826, 576, 89, 24)

    text("pose_label", "<b>YOLOv8-Pose</b>", 1015, 98, 170, 28, 13, True)
    image("pose_img", assets["tracknet"], 982, 128, 206, 112)
    text("det_label", "<b>Detections</b>", 1040, 270, 125, 28, 11, True)
    vertex("bytetrack", "<b>ByteTrack</b>", "shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;fillColor=#eaf2ff;strokeColor=#6c8dbf;fontSize=13;fontFamily=Helvetica;align=center;verticalAlign=middle;", 1010, 405, 165, 37)
    text("tracks_label", "<b>Player Tracks (ID)</b>", 1010, 451, 160, 28, 11, True)
    text("homography_label", "<b>Homography</b><br><b>(Court Mapping)</b>", 1232, 98, 150, 45, 13, True)
    rect("homography_court", 1238, 145, 140, 125, "#fbfbff", "#8581b4", True)
    text("h_label", "<b>H</b>", 1278, 282, 55, 25, 16, True)
    text("topdown_label", "<b>Top-down View</b>", 1235, 315, 150, 28, 12, True)
    rect("topdown_court", 1235, 342, 153, 108, "#f8fff8", "#777777")
    rect("motion_stats", 1225, 490, 167, 135, "#eaf4ff", "#7c9cc3", True)
    text("motion_label", "<b>Motion Statistics</b>", 1235, 497, 145, 28, 12, True)
    for i, label in enumerate(["Speed (m/s)", "Distance (m)", "Max Speed (m/s)", "..."]):
        vertex(f"motion_step_{i}", label, mx_style("#f8fbff", "#89a4c3", False, False, "fontSize=9;fontFamily=Helvetica;align=center;verticalAlign=middle;"), 1238, 525 + i * 25, 140, 21)

    text("overlay_label", "<b>Overlay Video</b>", 1495, 98, 170, 28, 13, True)
    image("overlay_img", assets["overlay"], 1472, 130, 206, 112)
    text("render_ellipsis", "...", 1555, 248, 70, 25, 13, True)
    text("final_label", "<b>Final Video</b>", 1495, 268, 170, 28, 13, True)
    image("final_img", assets["final"], 1472, 295, 206, 112)
    rect("data_export", 1460, 448, 230, 110, "#f4f2ff", "#8581b4", True)
    text("data_export_label", "<b>Data Export</b>", 1490, 455, 170, 28, 13, True)
    for cid, x, label, color in [("csv", 1474, "CSV", "#44a060"), ("json", 1548, "JSON", "#d0792d"), ("mp4", 1622, "MP4", "#6d4aa0")]:
        vertex(f"file_{cid}", f"<b>{label}</b>", mx_style(WHITE, color, True, False, f"fontSize=14;fontStyle=1;fontFamily=Helvetica;fontColor={color};align=center;verticalAlign=middle;"), x, 486, 54, 56)

    rect("dashboard_box", 1454, 620, 266, 272, PINK_FILL, "#e2a0aa", True)
    text("dashboard_title", "<b>Frontend Dashboard</b><br><b>(D3.js)</b>", 1490, 630, 175, 55, 15, True)
    image("dashboard_img", assets["dashboard"], 1465, 685, 245, 195)

    rect("bottom_outputs", 15, 665, 1390, 230, WHITE, "#1c2b44", False, True)
    for cid, x, title in [("ball", 25, "Ball Trajectory (CSV)"), ("players", 305, "Players (CSV)"), ("motion", 595, "Motion (CSV)"), ("stats", 875, "Stats (JSON)"), ("qual", 1152, "Quality (JSON)")]:
        rect(f"out_{cid}", x, 682, 265 if cid != "qual" else 240, 180, "#fafbff", "#b5bfd1", True)
        text(f"out_{cid}_title", f"<b>{title}</b>", x + 40, 690, 180, 28, 13, True)
    text("stats_json_text", '{<br>  "video": "01.mp4",<br>  "duration": 18.97,<br>  "players": {<br>    "1": {"distance": 78.34},<br>    "2": {"max_speed": %.2f}<br>  }<br>}' % analysis_meta.get("max_speed_mps", 9.46), 890, 715, 210, 120, 9, False, "left")
    text("quality_json_text", '{<br>  "video": "01.mp4",<br>  "quality_score": %.2f,<br>  "level": "Green",<br>  "visibility": 0.72,<br>  "max_gap": 4<br>}' % dash_score, 1168, 715, 160, 110, 9, False, "left")
    text("quality_json_gauge", f"<b>{dash_score:.2f}</b><br>/100<br><font color=\"#1f7a2f\">Green</font>", 1320, 730, 90, 110, 16, True)

    # Main flow edges.
    for cid, src, tgt, dashed in [
        ("e_input_det", "panel_1", "panel_2", False),
        ("e_det_ref", "panel_2", "panel_3", False),
        ("e_ref_player", "panel_3", "motion_stats", True),
        ("e_player_render", "panel_4", "panel_5", False),
        ("e_export_dashboard", "data_export", "dashboard_box", False),
    ]:
        edge(cid, src, tgt, dashed)

    # Legend.
    text("legend_data", "Data Flow", 95, 910, 100, 35, 11)
    text("legend_module", "Module Flow", 250, 910, 110, 35, 11)
    for i, (x, fill, label) in enumerate(
        [(385, BLUE_FILL, "Detection Module"), (585, GREEN_FILL, "Refinement Module"), (780, YELLOW_FILL, "Analysis Module"), (970, PINK_FILL, "Rendering/Export"), (1170, PURPLE_FILL, "Output Files")]
    ):
        rect(f"legend_box_{i}", x, 915, 38, 20, fill, "#8a9ab5")
        text(f"legend_label_{i}", label, x + 45, 902, 160, 35, 11, False, "left")

    ET.indent(mx, space="  ")
    ET.ElementTree(mx).write(DRAWIO, encoding="utf-8", xml_declaration=False)
    ET.parse(DRAWIO)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_png()
    make_svg()
    make_drawio()
    print(f"generated {DRAWIO}")
    print(f"generated {SVG}")
    print(f"generated {PNG}")


if __name__ == "__main__":
    main()
