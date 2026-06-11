from __future__ import annotations

import json
import math
import statistics
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent
ROOT_DIR = OUT_DIR.parents[1]
DRAWIO = OUT_DIR / "badminton_cvpr_pipeline.drawio"
SVG = OUT_DIR / "badminton_cvpr_pipeline.svg"
PNG = OUT_DIR / "badminton_cvpr_pipeline.png"

W, H = 2800, 1600

INK = "#111827"
MUTED = "#4b5563"
SOFT_TEXT = "#64748b"
LINE = "#263445"
WHITE = "#ffffff"
PANEL_STROKE = "#cbd5e1"
BLUE = "#4f75c8"
BLUE_DARK = "#3157a5"
BLUE_SOFT = "#edf4ff"
CYAN_SOFT = "#e8f7fb"
GREEN_SOFT = "#eaf8ef"
YELLOW_SOFT = "#fff8db"
ORANGE_SOFT = "#fff1df"
RED_SOFT = "#fff0f0"
LILAC_SOFT = "#f3f1ff"
GRAY_SOFT = "#f8fafc"


INPUTS = [
    ("video", "Match video", "RGB frames, fps, clip id", "required input"),
    ("court", "Court prior", "four corners + court size", "homography setup"),
    ("weights", "Model weights", "TrackNet + YOLOv8-pose", "local checkpoints"),
]

SHUTTLE_STAGES = [
    ("S1", "tensor", "Clip tensor", "3-frame stack for TrackNet", BLUE_SOFT),
    ("S2", "heatmap", "TrackNet heatmap", "shuttle probability map", BLUE_SOFT),
    ("S3", "filter", "Trajectory screening", "ROI, static lock, jump rules", GREEN_SOFT),
    ("S4", "smooth", "Temporal refinement", "Kalman smoothing + interpolation", GREEN_SOFT),
    ("S5", "quality", "Quality scoring", "visibility, gap, confidence", GREEN_SOFT),
]

PLAYER_STAGES = [
    ("P1", "pose", "YOLOv8-pose", "body keypoints per frame", ORANGE_SOFT),
    ("P2", "foot", "Foot-point extraction", "near/far player anchors", ORANGE_SOFT),
    ("P3", "track", "ByteTrack identity", "low-score association", YELLOW_SOFT),
    ("P4", "projection", "Court projection", "pixel plane to metric court", YELLOW_SOFT),
    ("P5", "speed", "Motion statistics", "distance, speed, rally time", YELLOW_SOFT),
]

OUTPUTS = [
    ("video_out", "Annotated video", "overlay.mp4 / final.mp4", "for visual review"),
    ("csv", "Structured tables", "ball.csv, players.csv, motion.csv", "for analysis"),
    ("json", "Quality report", "analysis.json, quality.json", "for audit"),
    ("dashboard", "Interactive dashboard", "mini-court, timeline, charts", "frontend export"),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def local_metrics() -> dict[str, str]:
    manifest_path = ROOT_DIR / "frontend/public/data/manifest.json"
    fallback = {
        "videos": "9",
        "frames": "4247",
        "duration": "150.68 s",
        "quality": "88.96-95.96",
        "mean_quality": "93.81",
        "visible": "69.19%",
        "distance": "714.72 m",
        "speed": "9.46 m/s",
        "level": "all Green",
    }
    if not manifest_path.exists():
        return fallback
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    videos = data.get("videos", [])
    if not videos:
        return fallback

    frames = sum(v.get("frame_count", 0) for v in videos)
    duration = sum(v.get("duration_s", 0.0) for v in videos)
    metas = [v.get("analysis_meta", {}) for v in videos]
    scores = [m.get("ball_quality_score") for m in metas if m.get("ball_quality_score") is not None]
    visible_rates = [m.get("ball_visible_rate") for m in metas if m.get("ball_visible_rate") is not None]
    distance = sum(m.get("near_distance_m", 0.0) + m.get("far_distance_m", 0.0) for m in metas)
    speeds = [m.get("max_speed_mps", 0.0) for m in metas]
    levels = sorted({m.get("ball_quality_level") for m in metas if m.get("ball_quality_level")})
    level = "all Green" if set(levels) == {"Green"} else "/".join(levels)

    return {
        "videos": str(len(videos)),
        "frames": f"{frames}",
        "duration": f"{duration:.2f} s",
        "quality": f"{min(scores):.2f}-{max(scores):.2f}" if scores else fallback["quality"],
        "mean_quality": f"{statistics.mean(scores):.2f}" if scores else fallback["mean_quality"],
        "visible": f"{statistics.mean(visible_rates) * 100:.2f}%" if visible_rates else fallback["visible"],
        "distance": f"{distance:.2f} m",
        "speed": f"{max(speeds):.2f} m/s" if speeds else fallback["speed"],
        "level": level,
    }


def icon_svg_body(name: str, stroke: str = INK) -> str:
    attrs = f'fill="none" stroke="{stroke}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"'
    fill = f'fill="{stroke}" stroke="none"'
    icons = {
        "video": f'<rect x="4" y="6" width="16" height="12" rx="2" {attrs}/><path d="M10 9l5 3-5 3z" {fill}/>',
        "court": f'<rect x="4" y="5" width="16" height="14" rx="1.5" {attrs}/><path d="M12 5v14M4 12h16M8 5v14M16 5v14" {attrs}/>',
        "weights": f'<circle cx="12" cy="12" r="3" {attrs}/><path d="M12 3v3M12 18v3M4.8 4.8l2.1 2.1M17.1 17.1l2.1 2.1M3 12h3M18 12h3M4.8 19.2l2.1-2.1M17.1 6.9l2.1-2.1" {attrs}/>',
        "tensor": f'<path d="M5 9l7-4 7 4-7 4zM5 9v6l7 4 7-4V9M12 13v6" {attrs}/>',
        "heatmap": f'<path d="M5 17c4-6 9-8 14-10" {attrs}/><circle cx="6" cy="17" r="2.2" {fill}/><path d="M16 5l3 2-2 3M14 6l3 2-2 3" {attrs}/>',
        "filter": f'<path d="M4 6h16l-6 7v5l-4 2v-7z" {attrs}/><path d="M7 6l5 7" {attrs}/>',
        "smooth": f'<path d="M4 15c3-5 5-5 8 0s5 5 8 0" {attrs}/><path d="M5 9c4-3 10-3 14 0" {attrs}/>',
        "quality": f'<circle cx="12" cy="12" r="8" {attrs}/><path d="M8.4 12.4l2.4 2.4 5-5.6" {attrs}/>',
        "pose": f'<circle cx="12" cy="5" r="2.4" {attrs}/><path d="M12 8v6M8 11h8M10 14l-3 6M14 14l3 6" {attrs}/>',
        "foot": f'<path d="M8 5c2 3 2 6-1 9l-2 2c2 2 5 1 7-1M15 6c-1 3 0 6 3 9l1 2c-2 2-5 1-7-1" {attrs}/>',
        "track": f'<circle cx="7" cy="8" r="2.2" {attrs}/><circle cx="17" cy="16" r="2.2" {attrs}/><path d="M9 9.5l6 5M6 17c3-5 7-8 12-10" {attrs}/>',
        "projection": f'<path d="M4 7l16-2v14L4 17zM4 12h16M9 6.4v11.2M15 5.6v12.8" {attrs}/>',
        "speed": f'<path d="M5 16a7 7 0 0 1 14 0" {attrs}/><path d="M12 16l4-6" {attrs}/><circle cx="12" cy="16" r="1.4" {fill}/>',
        "video_out": f'<rect x="4" y="5" width="16" height="14" rx="2" {attrs}/><path d="M8 5v14M16 5v14M4 10h4M16 10h4M4 15h4M16 15h4" {attrs}/>',
        "csv": f'<path d="M7 3h7l4 4v14H7zM14 3v5h5M10 12h6M10 16h6" {attrs}/>',
        "json": f'<path d="M8 8l-3 4 3 4M16 8l3 4-3 4M13 6l-2 12" {attrs}/>',
        "dashboard": f'<rect x="4" y="5" width="16" height="12" rx="2" {attrs}/><path d="M8 21h8M12 17v4M8 13l3-3 2 2 3-5" {attrs}/>',
    }
    return icons[name]


def icon_data_uri(name: str) -> str:
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">{icon_svg_body(name)}</svg>'
    return "data:image/svg+xml," + quote(svg, safe="")


def rounded(draw: ImageDraw.ImageDraw, box, radius, fill, outline=INK, width=3):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text_center(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, ft, fill=INK):
    bbox = draw.textbbox((0, 0), text, font=ft)
    draw.text((x - (bbox[2] - bbox[0]) / 2, y), text, font=ft, fill=fill)


def wrap_lines(text: str, chars: int) -> list[str]:
    return textwrap.wrap(text, width=chars, break_long_words=False)


def draw_arrow(draw: ImageDraw.ImageDraw, start, end, color=LINE, width=4, dashed=False):
    x1, y1 = start
    x2, y2 = end
    if dashed:
        steps = max(1, int(max(abs(x2 - x1), abs(y2 - y1)) // 20))
        for i in range(steps):
            if i % 2 == 0:
                a = i / steps
                b = min((i + 1) / steps, 1)
                draw.line(
                    [
                        (x1 + (x2 - x1) * a, y1 + (y2 - y1) * a),
                        (x1 + (x2 - x1) * b, y1 + (y2 - y1) * b),
                    ],
                    fill=color,
                    width=width,
                )
    else:
        draw.line([start, end], fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 16
    p1 = (x2, y2)
    p2 = (x2 - size * math.cos(angle - 0.45), y2 - size * math.sin(angle - 0.45))
    p3 = (x2 - size * math.cos(angle + 0.45), y2 - size * math.sin(angle + 0.45))
    draw.polygon([p1, p2, p3], fill=color)


def draw_segment(draw: ImageDraw.ImageDraw, start, end, color=LINE, width=4, dashed=False):
    x1, y1 = start
    x2, y2 = end
    if dashed:
        steps = max(1, int(max(abs(x2 - x1), abs(y2 - y1)) // 20))
        for i in range(steps):
            if i % 2 == 0:
                a = i / steps
                b = min((i + 1) / steps, 1)
                draw.line(
                    [
                        (x1 + (x2 - x1) * a, y1 + (y2 - y1) * a),
                        (x1 + (x2 - x1) * b, y1 + (y2 - y1) * b),
                    ],
                    fill=color,
                    width=width,
                )
    else:
        draw.line([start, end], fill=color, width=width)


def draw_poly_arrow(draw: ImageDraw.ImageDraw, points, color=LINE, width=4, dashed=False):
    for start, end in zip(points[:-2], points[1:-1]):
        draw_segment(draw, start, end, color=color, width=width, dashed=dashed)
    draw_arrow(draw, points[-2], points[-1], color=color, width=width, dashed=dashed)


def draw_icon(draw: ImageDraw.ImageDraw, name: str, x: int, y: int, size: int = 44, color: str = INK):
    ox, oy = x, y
    s = size / 24

    def pt(a, b):
        return (ox + a * s, oy + b * s)

    def line(points, width=3):
        draw.line([pt(a, b) for a, b in points], fill=color, width=width, joint="curve")

    def rect(a, b, c, d, width=3):
        draw.rounded_rectangle([*pt(a, b), *pt(c, d)], radius=4, outline=color, width=width)

    def circle(a, b, r, width=3, fill=None):
        draw.ellipse(
            [ox + (a - r) * s, oy + (b - r) * s, ox + (a + r) * s, oy + (b + r) * s],
            outline=color,
            width=width,
            fill=fill,
        )

    if name == "video":
        rect(4, 6, 20, 18)
        draw.polygon([pt(10, 9), pt(15, 12), pt(10, 15)], fill=color)
    elif name == "court":
        rect(4, 5, 20, 19)
        line([(12, 5), (12, 19)])
        line([(4, 12), (20, 12)])
        line([(8, 5), (8, 19)])
        line([(16, 5), (16, 19)])
    elif name == "weights":
        circle(12, 12, 3)
        for a, b, c, d in [(12, 3, 12, 6), (12, 18, 12, 21), (3, 12, 6, 12), (18, 12, 21, 12), (4.8, 4.8, 6.9, 6.9), (17.1, 17.1, 19.2, 19.2), (4.8, 19.2, 6.9, 17.1), (17.1, 6.9, 19.2, 4.8)]:
            line([(a, b), (c, d)])
    elif name == "tensor":
        line([(5, 9), (12, 5), (19, 9), (12, 13), (5, 9)])
        line([(5, 9), (5, 15), (12, 19), (19, 15), (19, 9)])
        line([(12, 13), (12, 19)])
    elif name == "heatmap":
        line([(5, 17), (10, 11), (15, 8), (19, 7)])
        circle(6, 17, 2.1, fill=color)
        line([(16, 5), (19, 7), (17, 10)])
        line([(14, 6), (17, 8), (15, 11)])
    elif name == "filter":
        line([(4, 6), (20, 6), (14, 13), (14, 18), (10, 20), (10, 13), (4, 6)])
        line([(7, 6), (12, 13)])
    elif name == "smooth":
        line([(4, 15), (8, 10), (12, 15), (16, 20), (20, 15)])
        line([(5, 9), (10, 7), (15, 7), (19, 9)])
    elif name == "quality":
        circle(12, 12, 8)
        line([(8.4, 12.4), (10.8, 14.8), (15.8, 9.2)])
    elif name == "pose":
        circle(12, 5, 2.4)
        line([(12, 8), (12, 14)])
        line([(8, 11), (16, 11)])
        line([(10, 14), (7, 20)])
        line([(14, 14), (17, 20)])
    elif name == "foot":
        line([(8, 5), (10, 9), (9, 13), (5, 16), (7, 18), (12, 15)])
        line([(15, 6), (14, 10), (16, 14), (19, 17), (17, 19), (12, 16)])
    elif name == "track":
        circle(7, 8, 2.2)
        circle(17, 16, 2.2)
        line([(9, 9.5), (15, 14)])
        line([(6, 17), (9, 13), (13, 10), (18, 7)])
    elif name == "projection":
        line([(4, 7), (20, 5), (20, 19), (4, 17), (4, 7)])
        line([(4, 12), (20, 12)])
        line([(9, 6.4), (9, 17.6)])
        line([(15, 5.6), (15, 18.4)])
    elif name == "speed":
        draw.arc([*pt(5, 9), *pt(19, 23)], 180, 360, fill=color, width=3)
        line([(12, 16), (16, 10)])
        circle(12, 16, 1.4, fill=color)
    elif name == "video_out":
        rect(4, 5, 20, 19)
        line([(8, 5), (8, 19)])
        line([(16, 5), (16, 19)])
        line([(4, 10), (8, 10)])
        line([(16, 10), (20, 10)])
        line([(4, 15), (8, 15)])
        line([(16, 15), (20, 15)])
    elif name == "csv":
        line([(7, 3), (14, 3), (18, 7), (18, 21), (7, 21), (7, 3)])
        line([(14, 3), (14, 8), (19, 8)])
        line([(10, 12), (16, 12)])
        line([(10, 16), (16, 16)])
    elif name == "json":
        line([(8, 8), (5, 12), (8, 16)])
        line([(16, 8), (19, 12), (16, 16)])
        line([(13, 6), (11, 18)])
    elif name == "dashboard":
        rect(4, 5, 20, 17)
        line([(8, 21), (16, 21)])
        line([(12, 17), (12, 21)])
        line([(8, 13), (11, 10), (13, 12), (16, 7)])


def draw_panel_label(draw, x, y, letter, title, ft_letter, ft_title):
    rounded(draw, [x, y, x + 55, y + 42], 8, BLUE, BLUE, 1)
    text_center(draw, x + 27.5, y + 8, letter, ft_letter, WHITE)
    draw.text((x + 68, y + 5), title, font=ft_title, fill=INK)


def draw_card(draw, x, y, w, h, fill, icon, title, detail, ft_title, ft_line, step=None, note=None):
    rounded(draw, [x, y, x + w, y + h], 12, fill, INK, 3)
    if step:
        rounded(draw, [x + 16, y + 15, x + 64, y + 43], 7, BLUE, BLUE, 1)
        text_center(draw, x + 40, y + 17, step, ft_line, WHITE)
        icon_x = x + 20
        icon_y = y + 63
    else:
        icon_x = x + 22
        icon_y = y + 45
    draw.ellipse([icon_x, icon_y, icon_x + 58, icon_y + 58], fill=WHITE, outline=INK, width=3)
    draw_icon(draw, icon, icon_x + 8, icon_y + 8, 42)
    tx = x + 92
    draw.text((tx, y + 28), title, font=ft_title, fill=INK)
    detail_chars = max(17, int((w - 108) / 8.4))
    for i, line in enumerate(wrap_lines(detail, detail_chars)):
        draw.text((tx, y + 66 + i * 25), line, font=ft_line, fill=MUTED)
    if note:
        draw.text((tx, y + h - 34), note, font=ft_line, fill=SOFT_TEXT)


def draw_chip(draw, x, y, text, fill, stroke, ft):
    bbox = draw.textbbox((0, 0), text, font=ft)
    w = bbox[2] - bbox[0] + 26
    rounded(draw, [x, y, x + w, y + 34], 9, fill, stroke, 2)
    draw.text((x + 13, y + 8), text, font=ft, fill=INK)
    return w


def draw_dashed_box(draw, box, color="#ef4444", width=3):
    x1, y1, x2, y2 = box
    dash = 14
    gap = 10
    for x in range(int(x1), int(x2), dash + gap):
        draw.line([(x, y1), (min(x + dash, x2), y1)], fill=color, width=width)
        draw.line([(x, y2), (min(x + dash, x2), y2)], fill=color, width=width)
    for y in range(int(y1), int(y2), dash + gap):
        draw.line([(x1, y), (x1, min(y + dash, y2))], fill=color, width=width)
        draw.line([(x2, y), (x2, min(y + dash, y2))], fill=color, width=width)


def make_png():
    metrics = local_metrics()
    img = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(img)

    ft_title = font(46, True)
    ft_sub = font(24)
    ft_panel = font(26, True)
    ft_letter = font(24, True)
    ft_card_title = font(23, True)
    ft_card_line = font(18)
    ft_small_title = font(21, True)
    ft_small = font(17)
    ft_formula = font(22, True)
    ft_caption = font(21)

    text_center(draw, W / 2, 32, "Figure 2. Proposed Badminton Video Analysis Workflow", ft_title, INK)
    text_center(draw, W / 2, 86, "Dual-branch perception, court-plane projection, quality screening, and visual analytics", ft_sub, MUTED)

    rounded(draw, [70, 170, 490, 1225], 16, WHITE, PANEL_STROKE, 3)
    rounded(draw, [520, 170, 2240, 1225], 16, WHITE, PANEL_STROKE, 3)
    rounded(draw, [2270, 170, 2730, 1225], 16, WHITE, PANEL_STROKE, 3)
    draw_panel_label(draw, 100, 205, "A", "Data and Prior Setup", ft_letter, ft_panel)
    draw_panel_label(draw, 550, 205, "B", "Dual-branch Perception Pipeline", ft_letter, ft_panel)
    draw_panel_label(draw, 2300, 205, "C", "Analytics Outputs", ft_letter, ft_panel)

    y_inputs = [330, 560, 790]
    for y, (icon, title, detail, note) in zip(y_inputs, INPUTS):
        draw_card(draw, 110, y, 350, 155, GRAY_SOFT, icon, title, detail, ft_card_title, ft_card_line, note=note)

    # Shuttle branch.
    rounded(draw, [560, 285, 2200, 610], 16, "#fbfdff", "#d7dee8", 2)
    draw.text((585, 302), "Shuttle trajectory branch", font=ft_small_title, fill=LINE)
    shuttle_x = [590, 910, 1230, 1550, 1870]
    for x, (step, icon, title, detail, fill) in zip(shuttle_x, SHUTTLE_STAGES):
        draw_card(draw, x, 360, 285, 145, fill, icon, title, detail, ft_card_title, ft_card_line, step=step)
    for x1, x2 in zip([875, 1195, 1515, 1835], [904, 1224, 1544, 1864]):
        draw_arrow(draw, (x1, 432), (x2, 432))

    # Screening cues inspired by paper-style dashed annotations.
    draw_dashed_box(draw, [1248, 522, 1815, 582], "#ef4444", 2)
    draw_chip(draw, 1270, 535, "ROI reject", RED_SOFT, "#fecaca", ft_small)
    draw_chip(draw, 1428, 535, "static lock", RED_SOFT, "#fecaca", ft_small)
    draw_chip(draw, 1585, 535, "jump rule", RED_SOFT, "#fecaca", ft_small)
    draw_chip(draw, 1735, 535, "interp", GREEN_SOFT, "#bbf7d0", ft_small)

    # Formula fusion ribbon.
    rounded(draw, [620, 645, 2150, 730], 13, CYAN_SOFT, "#9cc9da", 2)
    draw.text((650, 672), "Metric evidence fusion", font=ft_small_title, fill=INK)
    draw.text((970, 672), "p_c = H p_i", font=ft_formula, fill=LINE)
    draw.text((1245, 672), "v_t = ||p_t - p_{t-1}|| / dt", font=ft_formula, fill=LINE)
    draw.text((1620, 672), "Q = sum_i w_i s_i", font=ft_formula, fill=LINE)
    draw.text((1890, 672), "frontend export", font=ft_formula, fill=LINE)

    # Player branch.
    rounded(draw, [560, 765, 2200, 1088], 16, "#fffdf8", "#d7dee8", 2)
    draw.text((585, 782), "Player motion branch", font=ft_small_title, fill=LINE)
    player_x = [590, 910, 1230, 1550, 1870]
    for x, (step, icon, title, detail, fill) in zip(player_x, PLAYER_STAGES):
        draw_card(draw, x, 840, 285, 145, fill, icon, title, detail, ft_card_title, ft_card_line, step=step)
    for x1, x2 in zip([875, 1195, 1515, 1835], [904, 1224, 1544, 1864]):
        draw_arrow(draw, (x1, 912), (x2, 912))

    # Input connections and branch fusion.
    draw_arrow(draw, (460, 407), (584, 432))
    draw_arrow(draw, (460, 407), (584, 912))
    draw_arrow(draw, (460, 637), (1550, 645), dashed=True)
    draw_poly_arrow(draw, [(460, 867), (535, 720), (690, 505), (910, 432)], dashed=True)
    draw_arrow(draw, (460, 867), (590, 840), dashed=True)
    draw_arrow(draw, (2012, 505), (1980, 645))
    draw_arrow(draw, (1693, 840), (1693, 730))
    draw_arrow(draw, (2012, 985), (2020, 730))

    # Outputs.
    out_y = [315, 535, 755, 975]
    for y, (icon, title, detail, note) in zip(out_y, OUTPUTS):
        fill = LILAC_SOFT if icon == "dashboard" else GRAY_SOFT
        draw_card(draw, 2310, y, 390, 150, fill, icon, title, detail, ft_card_title, ft_card_line, note=note)
    draw_arrow(draw, (2145, 432), (2300, 390))
    draw_arrow(draw, (2145, 687), (2300, 610))
    draw_arrow(draw, (2145, 687), (2300, 830))
    draw_arrow(draw, (2145, 687), (2300, 1050))

    # Legend and evidence strip.
    rounded(draw, [340, 1270, 2460, 1354], 13, "#f8fafc", "#94a3b8", 2)
    evidence = (
        f"Verified local data: {metrics['videos']} videos | {metrics['frames']} frames | "
        f"{metrics['duration']} | {metrics['level']} | quality {metrics['quality']} "
        f"(mean {metrics['mean_quality']}) | shuttle visible {metrics['visible']} | "
        f"player distance {metrics['distance']} | max speed {metrics['speed']}"
    )
    text_center(draw, W / 2, 1298, evidence, font(19, True), LINE)

    legend_x = 540
    legend_y = 1385
    for label, fill, stroke in [
        ("perception module", BLUE_SOFT, "#9fb8e8"),
        ("filtering / quality", GREEN_SOFT, "#a7d7b5"),
        ("geometry / motion", YELLOW_SOFT, "#dcc56d"),
        ("prior or checkpoint", WHITE, "#94a3b8"),
    ]:
        if label == "prior or checkpoint":
            draw_dashed_box(draw, [legend_x, legend_y + 2, legend_x + 34, legend_y + 28], stroke, 2)
        else:
            rounded(draw, [legend_x, legend_y + 2, legend_x + 34, legend_y + 28], 4, fill, stroke, 2)
        draw.text((legend_x + 46, legend_y), label, font=ft_small, fill=MUTED)
        legend_x += 360

    caption = (
        "Fig. 2. Overview of the proposed workflow. The video stream is processed by a shuttle trajectory branch "
        "and a player motion branch; court priors project observations onto a metric court plane, after which "
        "quality-controlled evidence is exported as videos, tables, reports, and an interactive dashboard."
    )
    draw.multiline_text((300, 1450), "\n".join(wrap_lines(caption, 160)), font=ft_caption, fill=INK, spacing=7)
    img.save(PNG)


def make_svg():
    metrics = local_metrics()
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M2,2 L10,6 L2,10 z" fill="#263445"/></marker></defs>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}.title{font-size:46px;font-weight:700;fill:#111827}.sub{font-size:24px;fill:#4b5563}.panel{font-size:26px;font-weight:700;fill:#111827}.smallt{font-size:21px;font-weight:700;fill:#263445}.cardt{font-size:22px;font-weight:700;fill:#111827}.cardl{font-size:17px;fill:#4b5563}.formula{font-size:22px;font-weight:700;fill:#263445}.caption{font-size:21px;fill:#111827}</style>',
        f'<text x="{W/2}" y="70" text-anchor="middle" class="title">Figure 2. Proposed Badminton Video Analysis Workflow</text>',
        f'<text x="{W/2}" y="113" text-anchor="middle" class="sub">Dual-branch perception, court-plane projection, quality screening, and visual analytics</text>',
    ]

    def rect(x, y, w, h, r, fill, stroke=INK, sw=2, dash=False):
        dash_attr = ' stroke-dasharray="12 10"' if dash else ""
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>')

    def label(x, y, letter, title):
        rect(x, y, 55, 42, 8, BLUE, BLUE, 1)
        parts.append(f'<text x="{x+27.5}" y="{y+29}" text-anchor="middle" font-size="24" font-weight="700" fill="#ffffff">{letter}</text>')
        parts.append(f'<text x="{x+68}" y="{y+33}" class="panel">{title}</text>')

    def card(x, y, w, h, fill, icon, title, detail, step=None, note=None):
        rect(x, y, w, h, 12, fill, INK, 2.5)
        if step:
            rect(x + 16, y + 15, 48, 28, 7, BLUE, BLUE, 1)
            parts.append(f'<text x="{x+40}" y="{y+35}" text-anchor="middle" font-size="17" font-weight="700" fill="#ffffff">{step}</text>')
            ix, iy = x + 20, y + 63
        else:
            ix, iy = x + 22, y + 45
        parts.append(f'<circle cx="{ix+29}" cy="{iy+29}" r="29" fill="#ffffff" stroke="{INK}" stroke-width="2.5"/>')
        parts.append(f'<g transform="translate({ix+8},{iy+8}) scale(1.75)">{icon_svg_body(icon)}</g>')
        parts.append(f'<text x="{x+92}" y="{y+52}" class="cardt">{title}</text>')
        detail_chars = max(17, int((w - 108) / 8.4))
        for i, line in enumerate(wrap_lines(detail, detail_chars)):
            parts.append(f'<text x="{x+92}" y="{y+84+i*24}" class="cardl">{line}</text>')
        if note:
            parts.append(f'<text x="{x+92}" y="{y+h-24}" font-size="17" fill="#64748b">{note}</text>')

    def edge(x1, y1, x2, y2, dashed=False):
        dash = ' stroke-dasharray="12 10"' if dashed else ""
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{LINE}" stroke-width="4" marker-end="url(#arrow)"{dash}/>')

    rect(70, 170, 420, 1055, 16, WHITE, PANEL_STROKE, 3)
    rect(520, 170, 1720, 1055, 16, WHITE, PANEL_STROKE, 3)
    rect(2270, 170, 460, 1055, 16, WHITE, PANEL_STROKE, 3)
    label(100, 205, "A", "Data and Prior Setup")
    label(550, 205, "B", "Dual-branch Perception Pipeline")
    label(2300, 205, "C", "Analytics Outputs")

    for y, item in zip([330, 560, 790], INPUTS):
        card(110, y, 350, 155, GRAY_SOFT, item[0], item[1], item[2], note=item[3])

    rect(560, 285, 1640, 325, 16, "#fbfdff", "#d7dee8", 2)
    parts.append('<text x="585" y="326" class="smallt">Shuttle trajectory branch</text>')
    for x, item in zip([590, 910, 1230, 1550, 1870], SHUTTLE_STAGES):
        card(x, 360, 285, 145, item[4], item[1], item[2], item[3], step=item[0])
    for x1, x2 in zip([875, 1195, 1515, 1835], [904, 1224, 1544, 1864]):
        edge(x1, 432, x2, 432)
    rect(1248, 522, 567, 60, 9, "none", "#ef4444", 2, True)
    for x, label_text in [(1270, "ROI reject"), (1428, "static lock"), (1585, "jump rule"), (1735, "interp")]:
        fill = GREEN_SOFT if label_text == "interp" else RED_SOFT
        rect(x, 535, 126 if label_text != "static lock" else 134, 34, 9, fill, "#fecaca", 1.5)
        parts.append(f'<text x="{x+13}" y="558" font-size="17" fill="{INK}">{label_text}</text>')

    rect(620, 645, 1530, 85, 13, CYAN_SOFT, "#9cc9da", 2)
    parts.append('<text x="650" y="699" class="smallt">Metric evidence fusion</text>')
    parts.append('<text x="970" y="699" class="formula">p_c = H p_i</text>')
    parts.append('<text x="1245" y="699" class="formula">v_t = ||p_t - p_{t-1}|| / dt</text>')
    parts.append('<text x="1620" y="699" class="formula">Q = sum_i w_i s_i</text>')
    parts.append('<text x="1890" y="699" class="formula">frontend export</text>')

    rect(560, 765, 1640, 323, 16, "#fffdf8", "#d7dee8", 2)
    parts.append('<text x="585" y="806" class="smallt">Player motion branch</text>')
    for x, item in zip([590, 910, 1230, 1550, 1870], PLAYER_STAGES):
        card(x, 840, 285, 145, item[4], item[1], item[2], item[3], step=item[0])
    for x1, x2 in zip([875, 1195, 1515, 1835], [904, 1224, 1544, 1864]):
        edge(x1, 912, x2, 912)

    for args in [
        (460, 407, 584, 432, False),
        (460, 407, 584, 912, False),
        (460, 637, 1550, 645, True),
        (460, 867, 910, 360, True),
        (460, 867, 590, 840, True),
        (2012, 505, 1980, 645, False),
        (1693, 840, 1693, 730, False),
        (2012, 985, 2020, 730, False),
    ]:
        edge(*args)

    for y, item in zip([315, 535, 755, 975], OUTPUTS):
        card(2310, y, 390, 150, LILAC_SOFT if item[0] == "dashboard" else GRAY_SOFT, item[0], item[1], item[2], note=item[3])
    for args in [(2145, 432, 2300, 390, False), (2145, 687, 2300, 610, False), (2145, 687, 2300, 830, False), (2145, 687, 2300, 1050, False)]:
        edge(*args)

    rect(340, 1270, 2120, 84, 13, "#f8fafc", "#94a3b8", 2)
    evidence = (
        f"Verified local data: {metrics['videos']} videos | {metrics['frames']} frames | "
        f"{metrics['duration']} | {metrics['level']} | quality {metrics['quality']} "
        f"(mean {metrics['mean_quality']}) | shuttle visible {metrics['visible']} | "
        f"player distance {metrics['distance']} | max speed {metrics['speed']}"
    )
    parts.append(f'<text x="{W/2}" y="1321" text-anchor="middle" font-size="19" font-weight="700" fill="{LINE}">{evidence}</text>')
    caption = (
        "Fig. 2. Overview of the proposed workflow. The video stream is processed by a shuttle trajectory branch "
        "and a player motion branch; court priors project observations onto a metric court plane, after which "
        "quality-controlled evidence is exported as videos, tables, reports, and an interactive dashboard."
    )
    for i, line in enumerate(wrap_lines(caption, 160)):
        parts.append(f'<text x="300" y="{1475+i*30}" class="caption">{line}</text>')
    parts.append("</svg>")
    SVG.write_text("\n".join(parts), encoding="utf-8")


def add_geo(cell, x, y, w, h):
    ET.SubElement(cell, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})


def make_drawio():
    metrics = local_metrics()
    mx = ET.Element("mxfile", {"host": "app.diagrams.net", "agent": "Codex", "version": "24.7.17", "type": "device"})
    diagram = ET.SubElement(mx, "diagram", {"id": "badminton-cvpr-flowchart-2", "name": "Figure 2 CVPR Workflow"})
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

    def panel(cid, x, y, w, h):
        return vertex(cid, "", "rounded=1;whiteSpace=wrap;html=1;arcSize=4;fillColor=#ffffff;strokeColor=#cbd5e1;strokeWidth=2;", x, y, w, h)

    def label(cid, x, y, letter, title):
        vertex(
            cid + "_box",
            f"<b>{letter}</b>",
            f"rounded=1;whiteSpace=wrap;html=1;fillColor={BLUE};strokeColor={BLUE};fontColor=#ffffff;fontSize=16;fontStyle=1;align=center;verticalAlign=middle;",
            x,
            y,
            55,
            42,
        )
        vertex(cid + "_title", f"<b>{title}</b>", "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontFamily=Helvetica;fontSize=18;fontColor=#111827;", x + 68, y + 2, 390, 44)

    def card(cid, x, y, w, h, fill, icon, title, detail, step=None, note=None):
        step_html = f'<font style="font-size:11px;color:#3157a5"><b>{step}</b></font> ' if step else ""
        note_html = f'<br><font style="font-size:10px;color:#64748b">{note}</font>' if note else ""
        value = f'{step_html}<b>{title}</b><br><font style="font-size:11px;color:#4b5563">{detail}</font>{note_html}'
        vertex(
            cid,
            value,
            f"rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor={fill};strokeColor=#111827;strokeWidth=1.7;align=left;verticalAlign=middle;spacingLeft=76;fontFamily=Helvetica;fontSize=14;fontColor=#111827;",
            x,
            y,
            w,
            h,
        )
        icon_x = x + 20
        icon_y = y + 60 if step else y + 44
        vertex(cid + "_circle", "", "ellipse;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#111827;strokeWidth=1.5;", icon_x, icon_y, 58, 58)
        vertex(cid + "_icon", "", "shape=image;html=1;imageAspect=0;aspect=fixed;verticalLabelPosition=bottom;verticalAlign=top;image=" + icon_data_uri(icon) + ";", icon_x + 9, icon_y + 9, 40, 40)

    def edge(cid, src, tgt, dashed=False):
        style = "endArrow=block;html=1;rounded=0;strokeWidth=2.2;strokeColor=#263445;fontFamily=Helvetica;"
        if dashed:
            style += "dashed=1;dashPattern=8 6;"
        cell = ET.SubElement(root, "mxCell", {"id": cid, "value": "", "style": style, "edge": "1", "parent": "1", "source": src, "target": tgt})
        ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})

    vertex(
        "title",
        '<b>Figure 2. Proposed Badminton Video Analysis Workflow</b><br><font style="font-size:12px;color:#4b5563">Dual-branch perception, court-plane projection, quality screening, and visual analytics</font>',
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontFamily=Helvetica;fontSize=28;fontColor=#111827;",
        440,
        28,
        1920,
        100,
    )

    panel("panel_a", 70, 170, 420, 1055)
    panel("panel_b", 520, 170, 1720, 1055)
    panel("panel_c", 2270, 170, 460, 1055)
    label("a", 100, 205, "A", "Data and Prior Setup")
    label("b", 550, 205, "B", "Dual-branch Perception Pipeline")
    label("c", 2300, 205, "C", "Analytics Outputs")

    for y, item in zip([330, 560, 790], INPUTS):
        card(f"input_{item[0]}", 110, y, 350, 155, GRAY_SOFT, item[0], item[1], item[2], note=item[3])

    vertex("shuttle_bg", "<b>Shuttle trajectory branch</b>", "rounded=1;whiteSpace=wrap;html=1;fillColor=#fbfdff;strokeColor=#d7dee8;strokeWidth=1.3;align=left;verticalAlign=top;spacingLeft=20;spacingTop=8;fontFamily=Helvetica;fontSize=14;fontColor=#263445;", 560, 285, 1640, 325)
    for x, item in zip([590, 910, 1230, 1550, 1870], SHUTTLE_STAGES):
        card(item[1], x, 360, 285, 145, item[4], item[1], item[2], item[3], step=item[0])

    vertex("screening_box", "", "rounded=1;whiteSpace=wrap;html=1;arcSize=9;fillColor=none;strokeColor=#ef4444;strokeWidth=1.4;dashed=1;dashPattern=8 6;", 1248, 522, 567, 60)
    for cid, x, label_text, fill in [
        ("chip_roi", 1270, "ROI reject", RED_SOFT),
        ("chip_static", 1428, "static lock", RED_SOFT),
        ("chip_jump", 1585, "jump rule", RED_SOFT),
        ("chip_interp", 1735, "interp", GREEN_SOFT),
    ]:
        vertex(cid, label_text, f"rounded=1;whiteSpace=wrap;html=1;arcSize=9;fillColor={fill};strokeColor=#fecaca;fontFamily=Helvetica;fontSize=10;fontColor=#111827;align=center;verticalAlign=middle;", x, 535, 126, 34)

    vertex(
        "fusion",
        "<b>Metric evidence fusion</b>&nbsp;&nbsp;&nbsp; p_c = H p_i &nbsp;&nbsp; v_t = ||p_t-p_{t-1}|| / dt &nbsp;&nbsp; Q = sum_i w_i s_i &nbsp;&nbsp; frontend export",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#e8f7fb;strokeColor=#9cc9da;strokeWidth=1.4;align=center;verticalAlign=middle;fontFamily=Helvetica;fontSize=14;fontColor=#263445;",
        620,
        645,
        1530,
        85,
    )

    vertex("player_bg", "<b>Player motion branch</b>", "rounded=1;whiteSpace=wrap;html=1;fillColor=#fffdf8;strokeColor=#d7dee8;strokeWidth=1.3;align=left;verticalAlign=top;spacingLeft=20;spacingTop=8;fontFamily=Helvetica;fontSize=14;fontColor=#263445;", 560, 765, 1640, 323)
    for x, item in zip([590, 910, 1230, 1550, 1870], PLAYER_STAGES):
        card(item[1], x, 840, 285, 145, item[4], item[1], item[2], item[3], step=item[0])

    for i, (src, tgt) in enumerate(
        [
            ("tensor", "heatmap"),
            ("heatmap", "filter"),
            ("filter", "smooth"),
            ("smooth", "quality"),
            ("pose", "foot"),
            ("foot", "track"),
            ("track", "projection"),
            ("projection", "speed"),
        ],
        1,
    ):
        edge(f"pipe_{i}", src, tgt)
    edge("input_video_to_tensor", "input_video", "tensor")
    edge("input_video_to_pose", "input_video", "pose")
    edge("court_to_projection", "input_court", "projection", True)
    edge("court_to_fusion", "input_court", "fusion", True)
    edge("weights_to_heatmap", "input_weights", "heatmap", True)
    edge("weights_to_pose", "input_weights", "pose", True)
    edge("quality_to_fusion", "quality", "fusion")
    edge("projection_to_fusion", "projection", "fusion")
    edge("speed_to_fusion", "speed", "fusion")

    for y, item in zip([315, 535, 755, 975], OUTPUTS):
        card(f"out_{item[0]}", 2310, y, 390, 150, LILAC_SOFT if item[0] == "dashboard" else GRAY_SOFT, item[0], item[1], item[2], note=item[3])
    edge("fusion_to_video", "fusion", "out_video_out")
    edge("fusion_to_csv", "fusion", "out_csv")
    edge("fusion_to_json", "fusion", "out_json")
    edge("fusion_to_dashboard", "fusion", "out_dashboard")

    evidence = (
        f"Verified local data: {metrics['videos']} videos | {metrics['frames']} frames | "
        f"{metrics['duration']} | {metrics['level']} | quality {metrics['quality']} "
        f"(mean {metrics['mean_quality']}) | shuttle visible {metrics['visible']} | "
        f"player distance {metrics['distance']} | max speed {metrics['speed']}"
    )
    vertex("evidence", evidence, "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8fafc;strokeColor=#94a3b8;strokeWidth=1.4;align=center;verticalAlign=middle;fontFamily=Helvetica;fontSize=13;fontStyle=1;fontColor=#263445;", 340, 1270, 2120, 84)
    caption = (
        "Fig. 2. Overview of the proposed workflow. The video stream is processed by a shuttle trajectory branch "
        "and a player motion branch; court priors project observations onto a metric court plane, after which "
        "quality-controlled evidence is exported as videos, tables, reports, and an interactive dashboard."
    )
    vertex("caption", caption, "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=top;fontFamily=Helvetica;fontSize=14;fontColor=#111827;whiteSpace=wrap;", 300, 1440, 2200, 90)

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
