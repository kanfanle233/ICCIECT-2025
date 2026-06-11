# -*- coding: utf-8 -*-
"""
File: 002_清洗标准化与核心表生成.py
Purpose: 三类核心数据清洗与标准字段输出
Source notebook: 高考数据处理大修版本-checkpoint.ipynb
Execution order: keep numeric order 001 -> 010
"""

from pathlib import Path
from typing import List

import pandas as pd

# 统一的工程/数据目录，避免硬编码绝对路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from gaokao_recommender.paths import *

DATA_DIR = DATA_RAW_DIR
OUTPUT_DIR = DATA_PROCESSED_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATH_COMBO = DATA_DIR / "subject_combo_to_mbti.csv"
PATH_MAJOR = DATA_DIR / "2023上海专业分数线.txt"
PATH_SCORE = DATA_DIR / "2023年考生高考成绩分布表（上海市）.txt"

TARGET_YEAR = 2023


def _require_columns(df: pd.DataFrame, columns: List[str], dataset_name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} 缺少必要列: {missing}")


def _to_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[^\d.]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _save_clean(df: pd.DataFrame, filename: str) -> None:
    """只保存到 data/processed/，保持 data/raw/ 只读"""
    out_path = OUTPUT_DIR / filename
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存: {out_path}")


def clean_combo_table(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=",", encoding="utf-8-sig")

    rename_map = {
        "subject_combo": "选科组合",
        "combo": "选科组合",
        "mbti": "MBTI",
        "count": "人数",
    }
    df = raw.rename(columns=rename_map).copy()

    _require_columns(df, ["选科组合", "MBTI", "人数"], "subject_combo_to_mbti")

    df["选科组合"] = df["选科组合"].astype(str).str.strip()
    df["MBTI"] = df["MBTI"].astype(str).str.upper().str.strip()
    df["人数"] = _to_numeric_series(df["人数"]).fillna(0)

    df = df[(df["选科组合"] != "") & (df["MBTI"].str.fullmatch(r"[A-Z]{4}", na=False))]

    # 同一选科组合+MBTI 的重复记录按人数累加，避免简单去重丢失信息
    df = (
        df.groupby(["选科组合", "MBTI"], as_index=False, dropna=False)["人数"]
        .sum()
        .sort_values(["选科组合", "MBTI"])
    )
    df["人数"] = df["人数"].round(0).astype(int)

    return df.reset_index(drop=True)


def clean_major_table(path: Path, target_year: int = TARGET_YEAR) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    df = raw.dropna(axis=1, how="all").copy()

    _require_columns(df, ["年份", "院校名称", "专业名称", "最低分"], "2023上海专业分数线")

    df["年份"] = _to_numeric_series(df["年份"]).astype("Int64")
    df = df[df["年份"] == target_year].copy()

    for col in ["最低分", "最低位次", "最高分", "平均分"]:
        if col in df.columns:
            df[col] = _to_numeric_series(df[col])

    df["院校名称"] = df["院校名称"].astype(str).str.strip()
    df["专业名称"] = df["专业名称"].astype(str).str.strip()

    df = df.dropna(subset=["院校名称", "专业名称", "最低分"])
    df = df[(df["院校名称"] != "") & (df["专业名称"] != "")]

    # 同一院校-专业保留录取门槛更高的记录（最低分更高；同分时最低位次更靠前）
    sort_cols = ["院校名称", "专业名称", "最低分"]
    ascending = [True, True, False]
    if "最低位次" in df.columns:
        sort_cols.append("最低位次")
        ascending.append(True)

    df = (
        df.sort_values(sort_cols, ascending=ascending)
        .drop_duplicates(subset=["院校名称", "专业名称"], keep="first")
        .reset_index(drop=True)
    )

    return df


def clean_score_table(path: Path, target_year: int = TARGET_YEAR) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    df = raw.copy()

    # 兼容不同列名，统一为 分数/人数/累计人数
    rename_candidates = {
        "本段人数": "人数",
        "人数（本段）": "人数",
        "累计": "累计人数",
    }
    df = df.rename(columns=rename_candidates)

    if "分数" not in df.columns and len(df.columns) >= 3:
        df.columns = ["分数", "人数", "累计人数", *df.columns[3:]]

    _require_columns(df, ["分数", "人数"], "2023年考生高考成绩分布表")

    # 保留真正分数行，去掉备注与说明行
    score_str = (
        df["分数"]
        .astype(str)
        .str.replace("分及以上", "", regex=False)
        .str.replace("分", "", regex=False)
        .str.strip()
    )
    df = df[score_str.str.match(r"^\d+$", na=False)].copy()
    df["分数"] = pd.to_numeric(score_str[df.index], errors="coerce")
    df["人数"] = _to_numeric_series(df["人数"]).fillna(0)

    # 同一分数可能出现多行，先汇总再计算累计人数，避免原始累计列不一致
    df = (
        df.groupby("分数", as_index=False)["人数"]
        .sum()
        .sort_values("分数", ascending=False)
        .reset_index(drop=True)
    )

    df["累计人数"] = df["人数"].cumsum()
    total = float(df["人数"].sum())
    df["percentile"] = df["累计人数"] / total if total > 0 else 0.0
    df["年份"] = target_year

    df = df[["年份", "分数", "人数", "累计人数", "percentile"]]
    df[["分数", "人数", "累计人数", "年份"]] = df[["分数", "人数", "累计人数", "年份"]].astype(int)

    return df


def main() -> None:
    pd.set_option("display.max_columns", 50)

    print("开始预处理核心数据...")

    df_combo = clean_combo_table(PATH_COMBO)
    _save_clean(df_combo, "subject_combo_to_mbti_clean.csv")

    df_major = clean_major_table(PATH_MAJOR, TARGET_YEAR)
    _save_clean(df_major, "2023上海专业分数线_clean.csv")

    df_score = clean_score_table(PATH_SCORE, TARGET_YEAR)
    _save_clean(df_score, "2023年考生高考成绩分布表_clean.csv")
    _save_clean(df_score, "上海一分一段_2023_clean.csv")

    print("\n预处理完成，关键结果:")
    print(f"df_combo: {df_combo.shape}")
    print(f"df_major: {df_major.shape}")
    print(f"df_score: {df_score.shape}")
    print(f"分数范围: {df_score['分数'].min()} - {df_score['分数'].max()}")


if __name__ == "__main__":
    main()


