# -*- coding: utf-8 -*-
"""
File: 007_SVD推荐系统_向量化优化.py
Purpose: 推荐主流程（随机化SVD初始化 + BPR排序优化 + MMR重排）
Source notebook: 高考数据处理大修版本-checkpoint.ipynb
Execution order: keep numeric order 001 -> 010
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from sklearn.decomposition import TruncatedSVD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 0. 路径 ----------
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from gaokao_recommender.paths import *

DATA_DIR = DATA_RAW_DIR
OUTPUT_DIR = DATA_PROCESSED_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 1. 超参数 ----------
RANDOM_SEED = 42
TOP_N = 10
CANDIDATE_POOL = 80
EMBED_DIM = 48
BPR_EPOCHS = 35
SAMPLES_PER_EPOCH = 30000
LEARNING_RATE = 0.05
L2_REG = 1e-4
MMR_LAMBDA = 0.85


# ---------- 2. 工具函数 ----------
def _pick_col(df: pd.DataFrame, candidates: List[str], field_name: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"缺少字段 {field_name}，候选列: {candidates}，实际列: {list(df.columns)}")


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def _safe_topk(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    part = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-part, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    part = np.take_along_axis(part, order, axis=1)
    return idx, part


# ---------- 3. 数据读取与标准化 ----------
print("加载清洗数据...")
df_combo_raw = pd.read_csv(OUTPUT_DIR / "subject_combo_to_mbti_clean.csv", encoding="utf-8-sig")
df_items = pd.read_csv(OUTPUT_DIR / "2023上海专业分数线_with_PredictedMBTI.csv", encoding="utf-8-sig")

combo_col = _pick_col(df_combo_raw, ["combo", "选科组合", "subject_combo"], "选科组合")
mbti_col = _pick_col(df_combo_raw, ["mbti", "MBTI"], "MBTI")
count_col = _pick_col(df_combo_raw, ["count", "人数"], "人数")

users = df_combo_raw[[combo_col, mbti_col, count_col]].copy()
users.columns = ["combo", "mbti", "count"]
users["combo"] = users["combo"].astype(str).str.strip()
users["mbti"] = users["mbti"].astype(str).str.upper().str.strip()
users["count"] = pd.to_numeric(users["count"], errors="coerce").fillna(0).astype(int)
users = users[(users["combo"] != "") & (users["mbti"].str.fullmatch(r"[A-Z]{4}", na=False))].reset_index(drop=True)
users["uid"] = "u_" + users.index.astype(str)

item_required = ["院校名称", "专业名称", "Predicted_MBTI"]
for col in item_required:
    if col not in df_items.columns:
        raise KeyError(f"专业数据缺少必要列: {col}")

if "批次" not in df_items.columns:
    df_items["批次"] = "本科批"
if "最低分" not in df_items.columns:
    df_items["最低分"] = np.nan

items = df_items.copy()
items["Predicted_MBTI"] = items["Predicted_MBTI"].fillna("ISTJ").astype(str).str.upper().str.strip()
items["school_major"] = items["院校名称"].astype(str) + "_" + items["专业名称"].astype(str)
items = items.reset_index(drop=True)

print(f"用户数: {len(users)}, 专业数: {len(items)}")

# ---------- 4. 构造启发式先验评分（用于候选集与初始化） ----------
batch_difficulty = {
    "专科批": 1,
    "本科批": 3,
    "本科提前批": 4,
    "高职提前批": 2,
    "提前批": 4,
    "艺术类本科批": 3,
    "体育类本科批": 3,
}
max_difficulty = float(max(batch_difficulty.values()))
item_difficulty = (
    items["批次"].map(batch_difficulty).fillna(3).astype(np.float32).to_numpy() / max_difficulty
)

user_mbti = users["mbti"].to_numpy(dtype=object)
item_mbti = items["Predicted_MBTI"].to_numpy(dtype=object)

user_prefix = np.array([s[:2] for s in user_mbti], dtype=object)
item_prefix = np.array([s[:2] for s in item_mbti], dtype=object)

exact_match = user_mbti[:, None] == item_mbti[None, :]
prefix_match = user_prefix[:, None] == item_prefix[None, :]
mbti_matrix = np.where(exact_match, 1.0, np.where(prefix_match, 0.5, 0.0)).astype(np.float32)

base_scores = (mbti_matrix * (1.0 - item_difficulty[None, :]) * 4.0 + 1.0).astype(np.float32)

# ---------- 5. 使用随机化SVD做低秩初始化 ----------
# 参考 TruncatedSVD 文档中的 randomized solver（Halko 2009）
n_users, n_items = base_scores.shape
if n_users < 1 or n_items < 2:
    raise RuntimeError("用户或专业数量不足，无法进行BPR训练")

max_rank = min(n_users, n_items) - 1
if max_rank < 1:
    raise RuntimeError("样本规模过小，无法进行SVD初始化")

rank_cap = min(EMBED_DIM, max_rank)
svd = TruncatedSVD(
    n_components=rank_cap,
    algorithm="randomized",
    n_iter=10,
    n_oversamples=15,
    random_state=RANDOM_SEED,
)
user_factors = svd.fit_transform(base_scores).astype(np.float32)
item_factors = svd.components_.T.astype(np.float32)

if rank_cap < EMBED_DIM:
    user_pad = np.zeros((n_users, EMBED_DIM - rank_cap), dtype=np.float32)
    item_pad = np.zeros((n_items, EMBED_DIM - rank_cap), dtype=np.float32)
    user_factors = np.hstack([user_factors, user_pad])
    item_factors = np.hstack([item_factors, item_pad])

print(f"SVD初始化完成: rank={rank_cap}, explained_variance={svd.explained_variance_ratio_.sum():.4f}")

# ---------- 6. 构建隐式反馈正样本（每个用户取先验评分前CANDIDATE_POOL个） ----------
pool_k = min(CANDIDATE_POOL, n_items)
pos_idx, _ = _safe_topk(base_scores, pool_k)
pos_sets = [set(row.tolist()) for row in pos_idx]
pos_lists = [np.array(sorted(list(s)), dtype=np.int32) for s in pos_sets]

# ---------- 7. BPR 成对排序优化 ----------
rng = np.random.default_rng(RANDOM_SEED)
users_with_pos = np.array([u for u, s in enumerate(pos_lists) if len(s) > 0], dtype=np.int32)
if len(users_with_pos) == 0:
    raise RuntimeError("没有可用正样本，无法训练BPR")

print("开始 BPR 训练...")
for epoch in range(1, BPR_EPOCHS + 1):
    epoch_loss = 0.0

    for _ in range(SAMPLES_PER_EPOCH):
        u = int(users_with_pos[rng.integers(0, len(users_with_pos))])
        i = int(pos_lists[u][rng.integers(0, len(pos_lists[u]))])

        j = int(rng.integers(0, n_items))
        while j in pos_sets[u]:
            j = int(rng.integers(0, n_items))

        u_vec = user_factors[u].copy()
        i_vec = item_factors[i].copy()
        j_vec = item_factors[j].copy()

        x_uij = float(np.dot(u_vec, i_vec - j_vec))
        x_uij = float(np.clip(x_uij, -20.0, 20.0))
        sigm = 1.0 / (1.0 + np.exp(-x_uij))
        grad = 1.0 - sigm

        user_factors[u] = u_vec + LEARNING_RATE * (grad * (i_vec - j_vec) - L2_REG * u_vec)
        item_factors[i] = i_vec + LEARNING_RATE * (grad * u_vec - L2_REG * i_vec)
        item_factors[j] = j_vec + LEARNING_RATE * (-grad * u_vec - L2_REG * j_vec)

        epoch_loss += -np.log(sigm + 1e-12)

    if epoch % 5 == 0 or epoch == 1:
        avg_loss = epoch_loss / SAMPLES_PER_EPOCH
        print(f"Epoch {epoch:02d}/{BPR_EPOCHS}, avg_bpr_loss={avg_loss:.4f}")

# ---------- 8. 批量Top-K检索 + MMR重排 ----------
item_norm = _normalize_rows(item_factors)
score_matrix = user_factors @ item_factors.T

candidate_k = min(50, n_items)
cand_idx, cand_scores = _safe_topk(score_matrix, candidate_k)


def mmr_select(user_candidate_idx: np.ndarray, user_candidate_scores: np.ndarray) -> List[int]:
    selected_positions: List[int] = []
    remaining = list(range(len(user_candidate_idx)))

    while remaining and len(selected_positions) < TOP_N:
        if not selected_positions:
            first_pos = int(remaining[int(np.argmax(user_candidate_scores[remaining]))])
            selected_positions.append(first_pos)
            remaining.remove(first_pos)
            continue

        best_pos = remaining[0]
        best_score = -1e18

        chosen_item_idx = user_candidate_idx[selected_positions]
        chosen_vecs = item_norm[chosen_item_idx]

        for pos in remaining:
            idx_item = int(user_candidate_idx[pos])
            relevance = float(user_candidate_scores[pos])
            similarity = float(np.max(chosen_vecs @ item_norm[idx_item]))
            mmr_score = MMR_LAMBDA * relevance - (1.0 - MMR_LAMBDA) * similarity
            if mmr_score > best_score:
                best_score = mmr_score
                best_pos = pos

        selected_positions.append(best_pos)
        remaining.remove(best_pos)

    return selected_positions


final_idx = np.zeros((n_users, TOP_N), dtype=np.int32)
final_scores = np.zeros((n_users, TOP_N), dtype=np.float32)

for u in range(n_users):
    selected_pos = mmr_select(cand_idx[u], cand_scores[u])
    selected_items = cand_idx[u][selected_pos]
    selected_scores = cand_scores[u][selected_pos]

    order = np.argsort(-selected_scores)
    selected_items = selected_items[order]
    selected_scores = selected_scores[order]

    fill_len = len(selected_items)
    final_idx[u, :fill_len] = selected_items
    final_scores[u, :fill_len] = selected_scores

# ---------- 9. 导出推荐结果 ----------
item_ids = items["school_major"].to_numpy(dtype=object)
out = pd.DataFrame(
    {
        "uid": np.repeat(users["uid"].to_numpy(dtype=object), TOP_N),
        "rank": np.tile(np.arange(1, TOP_N + 1), n_users),
        "school_major": item_ids[final_idx.reshape(-1)],
        "est_score": final_scores.reshape(-1),
    }
)

out = out.merge(users[["uid", "combo", "mbti"]], on="uid", how="left")
out = out.merge(
    items[["school_major", "院校名称", "专业名称", "最低分", "Predicted_MBTI", "批次"]],
    on="school_major",
    how="left",
)
out = out[["uid", "combo", "mbti", "rank", "院校名称", "专业名称", "最低分", "批次", "Predicted_MBTI", "est_score"]]

out_main = OUTPUT_DIR / "combo_based_recommendations_sklearn.csv"
out_bpr = OUTPUT_DIR / "combo_based_recommendations_bpr.csv"
out.to_csv(out_main, index=False, encoding="utf-8-sig")
out.to_csv(out_bpr, index=False, encoding="utf-8-sig")
print(f"已保存推荐结果: {out_main}")
print(f"已额外保存: {out_bpr}")

# ---------- 10. 导出候选集（供可解释性脚本复用） ----------
users_small = users[["uid", "combo", "mbti"]].copy()
items_small = items[["school_major", "院校名称", "专业名称", "Predicted_MBTI", "最低分", "批次"]].copy()

cand = users_small.assign(_k=1).merge(items_small.assign(_k=1), on="_k").drop(columns="_k")

cand_user_mbti = cand["mbti"].fillna("").astype(str)
cand_item_mbti = cand["Predicted_MBTI"].fillna("").astype(str)

cand_exact = cand_user_mbti.eq(cand_item_mbti)
cand_prefix = cand_user_mbti.str[:2].eq(cand_item_mbti.str[:2])
cand["mbti_w"] = np.select([cand_exact, cand_prefix], [1.0, 0.5], default=0.0).astype(np.float32)
cand["difficulty"] = cand["批次"].map(batch_difficulty).fillna(3).astype(np.float32) / max_difficulty
cand["rating"] = (cand["mbti_w"] * (1.0 - cand["difficulty"]) * 4.0 + 1.0).astype(np.float32)

cand_file = OUTPUT_DIR / "recommendation_candidates.csv"
cand.to_csv(cand_file, index=False, encoding="utf-8-sig")
print(f"已保存候选集: {cand_file}")

# ---------- 11. 分析图与报告 ----------
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
sns.histplot(base_scores.reshape(-1), bins=30, kde=True)
plt.title("先验评分分布")
plt.xlabel("base_score")

plt.subplot(2, 2, 2)
sns.histplot(score_matrix.reshape(-1), bins=30, kde=True)
plt.title("BPR训练后打分分布")
plt.xlabel("bpr_score")

plt.subplot(2, 2, 3)
sns.histplot(out["est_score"], bins=20, kde=True)
plt.title("最终Top-N分数分布")
plt.xlabel("est_score")

plt.subplot(2, 2, 4)
combo_mean = out.groupby("combo")["est_score"].mean().sort_values(ascending=False)
combo_mean.plot(kind="bar", color="steelblue")
plt.title("各选科组合平均推荐分数")
plt.xlabel("combo")
plt.ylabel("mean_score")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
fig_file = FIGURE_DIR / "recommendation_analysis_bpr.png"
legacy_fig = FIGURE_DIR / "recommendation_analysis_sklearn.png"
plt.savefig(fig_file, dpi=300)
plt.savefig(legacy_fig, dpi=300)
plt.close()

report_file = PROJECT_ROOT / "reports" / "recommendation_analysis_report_bpr.txt"
legacy_report = PROJECT_ROOT / "reports" / "recommendation_analysis_report_sklearn.txt"
with open(report_file, "w", encoding="utf-8") as f:
    f.write("=== 推荐系统分析报告 (BPR + MMR) ===\n\n")
    f.write(f"用户数量: {n_users}\n")
    f.write(f"专业数量: {n_items}\n")
    f.write(f"总推荐数量: {len(out)}\n")
    f.write(f"SVD初始化维度: {rank_cap}\n")
    f.write(f"SVD解释方差比: {svd.explained_variance_ratio_.sum():.4f}\n")
    f.write(f"BPR训练轮数: {BPR_EPOCHS}\n")
    f.write(f"BPR每轮采样数: {SAMPLES_PER_EPOCH}\n")
    f.write(f"MMR_lambda: {MMR_LAMBDA}\n\n")
    f.write("按选科组合的平均推荐分数:\n")
    f.write(combo_mean.to_string())

with open(legacy_report, "w", encoding="utf-8") as f_sync:
    with open(report_file, "r", encoding="utf-8") as f_src:
        f_sync.write(f_src.read())

print(f"分析图已保存: {fig_file}")
print(f"兼容图已保存: {legacy_fig}")
print(f"分析报告已保存: {report_file}")
print(f"兼容报告已保存: {legacy_report}")
print("推荐核心优化完成。")






