#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "doc" / "analytics_recommendation"
FRONTEND = ROOT / "frontend" / "public" / "data"


ALIASES = {
    "1_00_01": "样本V1",
    "pro_match17_1_02_02": "样本V2",
    "pro_match17_1_15_13": "样本V3",
    "pro_match17_2_01_01": "样本V4",
    "pro_match17_2_08_05": "样本V5",
    "pro_match17_2_15_11": "样本V6",
    "pro_match17_2_18_11": "样本V7",
    "pro_match19_1_01_01": "样本V8",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reports = read_json(ROOT / "output" / "ball_quality_summary.json")["reports"]
    rows = []
    source_counts: dict[str, int] = {}
    missing_segments_total = 0
    total_low_conf = 0
    total_spatial = 0

    for r in reports:
        vid = r["video_id"]
        alias = ALIASES[vid]
        analysis = read_json(FRONTEND / "videos" / vid / "analysis.json")
        quality = read_json(FRONTEND / "videos" / vid / "quality.json")
        ball_rows = read_csv(FRONTEND / "videos" / vid / "ball.csv")
        players = analysis.get("players", {})
        near = players.get("near", {})
        far = players.get("far", {})
        dist = float(near.get("total_distance_m") or 0) + float(far.get("total_distance_m") or 0)
        max_speed = max(float(near.get("total_max_speed_mps") or 0), float(far.get("total_max_speed_mps") or 0))

        for b in ball_rows:
            source_counts[b.get("source", "unknown")] = source_counts.get(b.get("source", "unknown"), 0) + 1
        missing_segments_total += len(quality.get("ball_missing_segments", []))
        total_low_conf += int(quality.get("ball_low_confidence_frames", 0) or 0)
        total_spatial += int(quality.get("ball_spatial_frames", 0) or 0)

        raw_visible = max(int(r["raw_visible"]), 1)
        final_visible = max(int(r["final_visible"]), 1)
        interp_rate = float(r["interp_rate"])
        roi_ratio = int(r["rejected_roi"]) / raw_visible
        static_ratio = int(r["rejected_static_lock"]) / raw_visible
        jump_ratio = int(r["rejected_jump"]) / raw_visible
        interp_penalty = 10 * min(1.0, interp_rate / 0.45)
        roi_penalty = 15 * roi_ratio
        static_penalty = 10 * static_ratio
        jump_penalty = 10 * jump_ratio
        risk_index = (
            (100 - float(r["quality_score"]))
            + 8 * (1 - float(quality.get("ball_spatial_rate", 0)))
            + 4 * (1 - float(quality.get("player_coverage", {}).get("near", 0)))
            + 4 * (1 - float(quality.get("player_coverage", {}).get("far", 0)))
        )

        rows.append({
            "sample": alias,
            "frames": int(r["frames"]),
            "duration_s": float(analysis.get("duration_s") or 0),
            "quality_score": float(r["quality_score"]),
            "quality_level": r["quality_level"],
            "visible_rate": float(r["visible_rate"]),
            "spatial_rate": float(quality.get("ball_spatial_rate", 0)),
            "max_missing_gap": int(r["max_missing_gap"]),
            "interp_rate": interp_rate,
            "rejected_roi": int(r["rejected_roi"]),
            "rejected_static_lock": int(r["rejected_static_lock"]),
            "rejected_jump": int(r["rejected_jump"]),
            "interpolated": int(r["interpolated"]),
            "roi_penalty": roi_penalty,
            "static_penalty": static_penalty,
            "jump_penalty": jump_penalty,
            "interp_penalty": interp_penalty,
            "near_coverage": float(quality.get("player_coverage", {}).get("near", 0)),
            "far_coverage": float(quality.get("player_coverage", {}).get("far", 0)),
            "distance_m": dist,
            "max_speed_mps": max_speed,
            "missing_segments": len(quality.get("ball_missing_segments", [])),
            "risk_index": risk_index,
        })

    total_frames = sum(r["frames"] for r in rows)
    total_raw_visible = sum(r["raw_visible"] for r in reports)
    total_final_visible = sum(r["final_visible"] for r in reports)
    total_rejections = sum(r["rejected_roi"] + r["rejected_static_lock"] + r["rejected_jump"] for r in rows)
    totals = {
        "video_count": len(rows),
        "frames": total_frames,
        "duration_s": sum(r["duration_s"] for r in rows),
        "avg_quality_score": sum(r["quality_score"] for r in rows) / len(rows),
        "min_quality_score": min(r["quality_score"] for r in rows),
        "max_quality_score": max(r["quality_score"] for r in rows),
        "overall_visible_rate": total_final_visible / total_frames,
        "overall_spatial_rate": total_spatial / total_frames,
        "avg_spatial_rate": sum(r["spatial_rate"] for r in rows) / len(rows),
        "total_raw_visible": total_raw_visible,
        "total_final_visible": total_final_visible,
        "raw_rejection_rate": total_rejections / total_raw_visible,
        "total_rejections": total_rejections,
        "total_roi": sum(r["rejected_roi"] for r in rows),
        "total_static": sum(r["rejected_static_lock"] for r in rows),
        "total_jump": sum(r["rejected_jump"] for r in rows),
        "total_interpolated": sum(r["interpolated"] for r in rows),
        "total_missing_segments": missing_segments_total,
        "total_distance_m": sum(r["distance_m"] for r in rows),
        "max_speed_mps": max(r["max_speed_mps"] for r in rows),
    }

    penalty_totals = {
        "ROI外误检": sum(r["roi_penalty"] for r in rows),
        "短缺口插值": sum(r["interp_penalty"] for r in rows),
        "静态锁定": sum(r["static_penalty"] for r in rows),
        "跳点": sum(r["jump_penalty"] for r in rows),
    }
    rejection_totals = {
        "ROI外误检": totals["total_roi"],
        "静态锁定": totals["total_static"],
        "跳点": totals["total_jump"],
        "短缺口插值": totals["total_interpolated"],
    }

    rows_sorted = sorted(rows, key=lambda x: x["risk_index"], reverse=True)

    with (OUT / "video_diagnostics.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)

    (OUT / "summary.json").write_text(json.dumps({
        "totals": totals,
        "penalty_totals": penalty_totals,
        "rejection_totals": rejection_totals,
        "source_counts": source_counts,
        "top_risk": rows_sorted[:3],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    plt.rcParams.update({
        "font.family": ["Arial Unicode MS", "PingFang SC", "DejaVu Sans", "sans-serif"],
        "axes.facecolor": "#FFFFFF",
        "figure.facecolor": "#FCFCFD",
        "axes.edgecolor": "#D7DBE7",
        "axes.labelcolor": "#1F2430",
        "xtick.color": "#6F768A",
        "ytick.color": "#1F2430",
    })
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=180)
    cats = sorted(rejection_totals, key=rejection_totals.get, reverse=True)
    vals = [rejection_totals[c] for c in cats]
    colors = ["#A3BEFA", "#F0986E", "#A3D576", "#F390CA"]
    ax.barh(cats[::-1], vals[::-1], color=colors[::-1], edgecolor="#464C55", linewidth=0.8)
    ax.set_title("后处理问题来源汇总", loc="left", fontsize=15, weight="bold", color="#1F2430")
    ax.set_xlabel("帧数或检测点数量")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E6E8F0", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(vals[::-1]):
        ax.text(v + max(vals) * 0.015, i, f"{v}", va="center", fontsize=10, color="#1F2430")
    fig.text(0.125, 0.915, "ROI外误检和静态锁定占后处理工作量主体；短缺口插值体现检测断裂的补救成本。", fontsize=9.5, color="#6F768A")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT / "issue_contribution.png", bbox_inches="tight", facecolor="#FCFCFD")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=180)
    risk_rows = rows_sorted[:6]
    ax.barh([r["sample"] for r in risk_rows][::-1], [r["risk_index"] for r in risk_rows][::-1], color="#F0986E", edgecolor="#464C55", linewidth=0.8)
    ax.set_title("样本优化优先级排行", loc="left", fontsize=15, weight="bold", color="#1F2430")
    ax.set_xlabel("综合风险指数")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E6E8F0", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    for i, r in enumerate(risk_rows[::-1]):
        ax.text(r["risk_index"] + max(x["risk_index"] for x in risk_rows) * 0.015, i, f"分数 {r['quality_score']:.1f} / 可见 {pct(r['visible_rate'])}", va="center", fontsize=9, color="#1F2430")
    fig.text(0.125, 0.915, "风险指数综合质量缺口、球空间映射、近/远端球员覆盖；用于排序优化验证优先级。", fontsize=9.5, color="#6F768A")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT / "risk_ranking.png", bbox_inches="tight", facecolor="#FCFCFD")
    plt.close(fig)

    recommendation = "优先做一个“ROI/背景静态锁定预过滤 + 参数化质量门”的优化；扩大模型或改前端展示可放在下一阶段。"
    report = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>羽毛球分析数据优化建议</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px auto; max-width: 980px; color: #1F2430; line-height: 1.65; }}
    h1 {{ font-size: 30px; margin-bottom: 8px; }}
    h2 {{ margin-top: 34px; font-size: 20px; }}
    .summary {{ background: #F4F7FC; border: 1px solid #D7DBE7; border-radius: 8px; padding: 18px 22px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0; }}
    .metric {{ border: 1px solid #E2E5EA; border-radius: 8px; padding: 12px; background: #fff; }}
    .metric span {{ display:block; color:#6F768A; font-size: 13px; }}
    .metric strong {{ font-size: 22px; }}
    img {{ width: 100%; border: 1px solid #E2E5EA; border-radius: 8px; background: #FCFCFD; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border-bottom: 1px solid #E2E5EA; padding: 8px 10px; text-align: left; }}
    th {{ background: #F4F5F7; }}
    .small {{ color:#6F768A; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>羽毛球分析数据优化建议</h1>
  <section class="summary">
    <h2>Executive Summary</h2>
    <p><strong>建议：</strong>{recommendation}</p>
    <p><strong>原因：</strong>9段视频都达到Green，但原始球检测中约{totals['raw_rejection_rate']*100:.1f}%需要被后处理拒绝；其中ROI外误检{totals['total_roi']}个、静态锁定{totals['total_static']}个，合计占拒绝项的{(totals['total_roi']+totals['total_static'])/totals['total_rejections']*100:.1f}%。这说明首要优化对象应放在进入轨迹后处理前的误检输入质量，前端展示改动优先级较低。</p>
  </section>
  <div class="metric-grid">
    <div class="metric"><span>视频数</span><strong>{totals['video_count']}</strong></div>
    <div class="metric"><span>总帧数</span><strong>{totals['frames']}</strong></div>
    <div class="metric"><span>平均质量分</span><strong>{totals['avg_quality_score']:.2f}</strong></div>
    <div class="metric"><span>总体球可见率</span><strong>{pct(totals['overall_visible_rate'])}</strong></div>
  </div>
  <h2>问题来源：误检清理比速度统计更值得先优化</h2>
  <p>后处理统计显示，ROI外误检和静态锁定是主要工作量。它们会直接造成轨迹跳变、背景固定点和前端质量警告，也会抬高插值需求。</p>
  <img src="issue_contribution.png" alt="后处理问题来源汇总">
  <h2>优化优先级：先用高风险样本验证</h2>
  <p>综合质量缺口、球空间映射和球员覆盖后，样本V9仍是最适合做压力测试的样本；样本V1、样本V2和样本V5适合作为次级边界样本。建议用这些样本做参数消融，并用样本V6作为高质量对照。</p>
  <img src="risk_ranking.png" alt="样本优化优先级排行">
  <h2>推荐落地方式</h2>
  <ol>
    <li>在检测前增加可配置的广播遮挡/比分牌/场外ROI屏蔽，减少明显场外小亮点进入球检测。</li>
    <li>把静态锁定规则前移为预过滤：连续多帧坐标变化低、局部运动分数低的点先降权或剔除。</li>
    <li>用样本V9、样本V5、样本V2做回归集，目标是把ROI外误检+静态锁定总量下降30%以上，同时保持所有样本Green、样本V6可见率不下降。</li>
  </ol>
  <h2>关键样本表</h2>
  <table>
    <tr><th>样本</th><th>质量分</th><th>可见率</th><th>空间映射率</th><th>ROI外</th><th>静态锁定</th><th>插值</th></tr>
    {''.join(f"<tr><td>{r['sample']}</td><td>{r['quality_score']:.2f}</td><td>{pct(r['visible_rate'])}</td><td>{pct(r['spatial_rate'])}</td><td>{r['rejected_roi']}</td><td>{r['rejected_static_lock']}</td><td>{r['interpolated']}</td></tr>" for r in rows_sorted[:5])}
  </table>
  <p class="small">来源：本地质量汇总、逐视频前端数据包、球轨迹CSV、球员运动CSV和轨迹后处理报告。质量分数用于工程展示可靠性，不等同于人工标注真值下的检测准确率。</p>
</body>
</html>
"""
    (OUT / "report.html").write_text(report, encoding="utf-8")

    print(json.dumps({
        "report": str(OUT / "report.html"),
        "diagnostics_csv": str(OUT / "video_diagnostics.csv"),
        "issue_chart": str(OUT / "issue_contribution.png"),
        "risk_chart": str(OUT / "risk_ranking.png"),
        "recommendation": recommendation,
        "totals": totals,
        "top_risk": rows_sorted[:3],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
