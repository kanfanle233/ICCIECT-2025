from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import cleaned_char_count, ensure_dir, load_all_splits, normalize_text, save_json


def quantiles(series: pd.Series) -> dict[str, float]:
    qs = series.quantile([0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    return {str(k): round(float(v), 2) for k, v in qs.items()}


def split_profile(df: pd.DataFrame) -> dict:
    text_norm = df["text_a"].map(normalize_text)
    clean_len = df["text_a"].map(cleaned_char_count)
    text_len = df["text_a"].map(lambda x: len(normalize_text(x)))
    labels = pd.to_numeric(df["label"], errors="coerce")
    bad_labels = sorted(set(labels.dropna().astype(int).unique()) - {0, 1})
    return {
        "rows": int(len(df)),
        "label_counts": {str(k): int(v) for k, v in labels.value_counts(dropna=False).sort_index().items()},
        "bad_labels": [int(x) for x in bad_labels],
        "null_text": int(df["text_a"].isna().sum()),
        "empty_text_after_strip": int((text_norm == "").sum()),
        "empty_text_after_cleaning": int((clean_len == 0).sum()),
        "exact_duplicate_rows": int(df.assign(text_norm=text_norm).duplicated(["label", "text_norm"]).sum()),
        "duplicate_text_rows": int(text_norm.duplicated().sum()),
        "unique_texts": int(text_norm.nunique(dropna=False)),
        "length_quantiles": quantiles(text_len),
    }


def build_report(frames: dict[str, pd.DataFrame], summary: dict, output_dir: Path) -> str:
    lines = [
        "# Data Quality Report",
        "",
        "Dataset: Chinese shopping comments sentiment classification.",
        "",
        "## Split Profiles",
        "",
    ]
    for split, profile in summary["split_profiles"].items():
        lines.extend(
            [
                f"### {split}",
                "",
                f"- Rows: {profile['rows']}",
                f"- Label counts: {profile['label_counts']}",
                f"- Bad labels: {profile['bad_labels']}",
                f"- Null text: {profile['null_text']}",
                f"- Empty after strip: {profile['empty_text_after_strip']}",
                f"- Empty after cleaning: {profile['empty_text_after_cleaning']}",
                f"- Exact duplicate rows: {profile['exact_duplicate_rows']}",
                f"- Duplicate text rows: {profile['duplicate_text_rows']}",
                f"- Text length quantiles: {profile['length_quantiles']}",
                "",
            ]
        )

    lines.extend(["## Cross-Split Overlap", ""])
    for item in summary["cross_split_overlap"]:
        lines.append(
            f"- {item['left']} vs {item['right']}: {item['overlap_unique_texts']} unique overlapping texts "
            f"({item['rate_vs_right_unique']:.4%} of {item['right']} unique texts)"
        )

    lines.extend(["", "## Conflicting Labels", ""])
    lines.append(f"- Unique normalized texts with conflicting labels: {summary['conflicting_label_texts']}")
    if summary["conflicting_label_texts"]:
        lines.append(f"- Sample saved to `{output_dir / 'conflicting_label_samples.csv'}`")

    lines.extend(["", "## Recommended Checks", ""])
    lines.extend(
        [
            "- Drop invalid labels, null text, and text that becomes empty after cleaning before modeling.",
            "- Keep train/dev/test fixed for fair comparison with the original notebooks.",
            "- Consider removing exact duplicate training rows and documenting cross-split overlaps before final reporting.",
            "- Inspect conflicting labels because they can cap the apparent performance of stronger NLP models.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check data quality for shopping comment sentiment data.")
    parser.add_argument("--data_dir", default="shopping_comments", help="Directory containing train/dev/test txt files.")
    parser.add_argument("--output_dir", default="outputs/data_quality", help="Directory for quality reports.")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    frames = load_all_splits(args.data_dir)

    enriched: dict[str, pd.DataFrame] = {}
    for split, df in frames.items():
        out = df.copy()
        out["text_norm"] = out["text_a"].map(normalize_text)
        out["text_len"] = out["text_a"].map(lambda x: len(normalize_text(x)))
        out["clean_len"] = out["text_a"].map(cleaned_char_count)
        enriched[split] = out

    summary = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "split_profiles": {split: split_profile(df) for split, df in frames.items()},
        "cross_split_overlap": [],
        "conflicting_label_texts": 0,
    }

    for left, right in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        left_texts = set(enriched[left]["text_norm"])
        right_texts = set(enriched[right]["text_norm"])
        overlap = left_texts & right_texts
        right_unique = max(1, enriched[right]["text_norm"].nunique(dropna=False))
        summary["cross_split_overlap"].append(
            {
                "left": left,
                "right": right,
                "overlap_unique_texts": int(len(overlap)),
                "rate_vs_right_unique": float(len(overlap) / right_unique),
            }
        )

    all_rows = pd.concat(enriched.values(), ignore_index=True)
    conflict_counts = all_rows.groupby("text_norm")["label"].nunique(dropna=False)
    conflict_texts = conflict_counts[conflict_counts > 1].index
    summary["conflicting_label_texts"] = int(len(conflict_texts))

    if len(conflict_texts):
        sample = all_rows[all_rows["text_norm"].isin(conflict_texts)][["split", "label", "text_a", "text_norm"]]
        sample.head(200).to_csv(output_dir / "conflicting_label_samples.csv", index=False, encoding="utf-8-sig")

    top_duplicates = (
        all_rows["text_norm"]
        .value_counts()
        .rename_axis("text_norm")
        .reset_index(name="count")
        .query("count > 1")
        .head(200)
    )
    top_duplicates.to_csv(output_dir / "top_duplicate_texts.csv", index=False, encoding="utf-8-sig")

    save_json(summary, output_dir / "data_quality_summary.json")
    report = build_report(enriched, summary, output_dir)
    (output_dir / "data_quality_report.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"Saved: {output_dir / 'data_quality_report.md'}")
    print(f"Saved: {output_dir / 'data_quality_summary.json'}")


if __name__ == "__main__":
    main()

