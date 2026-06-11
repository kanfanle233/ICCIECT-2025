"""
高考志愿填报辅助系统 - 基于SVD矩阵分解的推荐系统（sklearn版本）

使用 scikit-learn 的 TruncatedSVD 对用户-专业评分矩阵进行降维分解，
为每个选科组合生成 Top-10 专业推荐，并输出统计分析报告。
教学重点：矩阵分解推荐算法的向量化实现与性能对比。
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

print("开始构建推荐系统（使用 scikit-learn SVD）...")

# ---------- 1. 路径 ----------
BASE_DIR = Path("E:/代码/minicondapythonProject1")
DATA_DIR = BASE_DIR / "论文代码" / "志愿填报辅助系统" / "上海高考录取数据17-23年"

# ---------- 2. 读取数据 ----------
# 读取选科组合数据
df_combo = pd.read_csv(DATA_DIR / "subject_combo_to_mbti_clean.csv", encoding="utf-8-sig")
df_combo = df_combo.rename(columns={"subject_combo": "combo", "count": "count"})

# 读取专业MBTI数据
df_items = pd.read_csv(DATA_DIR / "2023上海专业分数线_with_PredictedMBTI.csv", encoding="utf-8-sig")
df_items["school_major"] = df_items["院校名称"] + "_" + df_items["专业名称"]

print(f"选科组合数据形状: {df_combo.shape}")
print(f"专业数据形状: {df_items.shape}")

# ---------- 3. 学生(pseudo-user) 画像 ----------
df_users = df_combo[["combo", "mbti", "count"]].copy()
df_users["uid"] = "u_" + df_users.index.astype(str)

print(f"用户数据形状: {df_users.shape}")
print(f"用户MBTI分布:\n{df_users['mbti'].value_counts()}")

# ---------- 4. 构造 MBTI 匹配度函数 ----------
def mbti_weight(stu_mbti, maj_mbti):
    if stu_mbti == maj_mbti:           # 完全一致
        return 1.0
    if stu_mbti[:2] == maj_mbti[:2]:   # 中间两位相同（NT/NF/ST/SF）
        return 0.5
    return 0.0

# ---------- 5. 生成伪评分表 ----------
# 创建用户-项目评分矩阵
user_ids = df_users["uid"].tolist()
item_ids = df_items["school_major"].tolist()

# 批次难度映射
batch_difficulty = {
    "专科批": 1,
    "本科批": 3,
    "本科提前批": 4,
    "高职提前批": 2,
    "提前批": 4,
    "艺术类本科批": 3,
    "体育类本科批": 3,
}

print("批次类别:", df_items["批次"].unique())

max_difficulty = float(max(batch_difficulty.values()))
item_difficulty = (
    df_items["批次"].map(batch_difficulty).fillna(3).astype(np.float32).to_numpy() / max_difficulty
)

user_mbti = df_users["mbti"].fillna("").astype(str).to_numpy()
item_mbti = df_items["Predicted_MBTI"].fillna("").astype(str).to_numpy()

user_prefix = np.array([s[:2] for s in user_mbti], dtype=object)
item_prefix = np.array([s[:2] for s in item_mbti], dtype=object)

exact_match = user_mbti[:, None] == item_mbti[None, :]
prefix_match = user_prefix[:, None] == item_prefix[None, :]

mbti_matrix = np.where(exact_match, 1.0, np.where(prefix_match, 0.5, 0.0)).astype(np.float32)
rating_matrix = (mbti_matrix * (1.0 - item_difficulty[None, :]) * 4.0 + 1.0).astype(np.float32)

print(f"评分矩阵形状: {rating_matrix.shape}")
print(f"评分范围: {rating_matrix.min():.2f} - {rating_matrix.max():.2f}")

# ---------- 6. 使用 TruncatedSVD 进行矩阵分解 ----------
n_components = min(50, min(rating_matrix.shape) - 1)  # 确保不超过矩阵维度
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_factors = svd.fit_transform(rating_matrix)
item_factors = svd.components_.T

print(f"SVD分解完成: {n_components} 个成分")
print(f"解释方差比: {svd.explained_variance_ratio_.sum():.3f}")

# 重建评分矩阵
reconstructed_ratings = user_factors @ item_factors.T

print(f"重建评分矩阵形状: {reconstructed_ratings.shape}")

# ---------- 7. Top-N 推荐函数 ----------
def recommend(user_index, N=10):
    user_scores = reconstructed_ratings[user_index]
    top_indices = np.argsort(user_scores)[::-1][:N]
    return [(item_ids[j], user_scores[j]) for j in top_indices]

# ---------- 8. 批量输出每个选科组合的 Top-10 ----------
TOP_N = 10
item_ids_arr = np.array(item_ids, dtype=object)

# 向量化提速：一次性取每个用户的 Top-N
raw_top_idx = np.argpartition(-reconstructed_ratings, kth=TOP_N - 1, axis=1)[:, :TOP_N]
raw_top_scores = np.take_along_axis(reconstructed_ratings, raw_top_idx, axis=1)
order = np.argsort(-raw_top_scores, axis=1)
top_idx = np.take_along_axis(raw_top_idx, order, axis=1)
top_scores = np.take_along_axis(raw_top_scores, order, axis=1)

all_recs = pd.DataFrame({
    "uid": np.repeat(np.array(user_ids, dtype=object), TOP_N),
    "rank": np.tile(np.arange(1, TOP_N + 1), len(user_ids)),
    "school_major": item_ids_arr[top_idx.reshape(-1)],
    "est_score": top_scores.reshape(-1)
})

out = all_recs

# 添加用户信息和专业信息
out = out.merge(df_users[["uid", "combo", "mbti"]], on="uid")
out = out.merge(df_items[["school_major", "院校名称", "专业名称", "最低分", "Predicted_MBTI"]], on="school_major")

# 重新排列列顺序
out = out[["uid", "combo", "mbti", "rank", "院校名称", "专业名称", "最低分", "Predicted_MBTI", "est_score"]]

# 保存结果
out.to_csv(DATA_DIR / "combo_based_recommendations_sklearn.csv", index=False, encoding="utf-8-sig")
print("✅ 推荐结果已保存 → combo_based_recommendations_sklearn.csv")

# ---------- 9. 查看示例 ----------
print("\n📋 推荐结果示例 (前20行):")
print(out.head(20))

# ---------- 10. 可视化分析 ----------
plt.figure(figsize=(15, 10))

# 原始评分分布
plt.subplot(2, 3, 1)
sns.histplot(rating_matrix.flatten(), bins=20, kde=True)
plt.title('原始评分分布')
plt.xlabel('评分')
plt.ylabel('数量')

# 重建评分分布
plt.subplot(2, 3, 2)
sns.histplot(reconstructed_ratings.flatten(), bins=20, kde=True)
plt.title('重建评分分布')
plt.xlabel('重建评分')
plt.ylabel('数量')

# 最终推荐分数分布
plt.subplot(2, 3, 3)
sns.histplot(out['est_score'], bins=20, kde=True)
plt.title('最终推荐分数分布')
plt.xlabel('推荐分数')
plt.ylabel('推荐数量')

# 按选科组合的平均分数
plt.subplot(2, 3, 4)
combo_scores = out.groupby('combo')['est_score'].mean().sort_values(ascending=False)
combo_scores.plot(kind='bar', color='skyblue')
plt.title('各选科组合平均推荐分数')
plt.xlabel('选科组合')
plt.ylabel('平均推荐分数')
plt.xticks(rotation=45)

# MBTI匹配分析
plt.subplot(2, 3, 5)
mbti_match = out.groupby(['mbti', 'Predicted_MBTI']).size().reset_index(name='count')
top_matches = mbti_match.nlargest(10, 'count')
sns.barplot(data=top_matches, x='count', y='mbti', hue='Predicted_MBTI')
plt.title('Top MBTI匹配组合')
plt.xlabel('匹配数量')

plt.tight_layout()
plt.savefig(DATA_DIR / "recommendation_analysis_sklearn.png", dpi=300)
plt.show()

# ---------- 11. 详细统计 ----------
print(f"\n🎯 推荐系统统计:")
print(f"用户数量: {len(user_ids)}")
print(f"专业数量: {len(item_ids)}")
print(f"总推荐数量: {len(out)}")
print(f"平均推荐分数: {out['est_score'].mean():.3f} ± {out['est_score'].std():.3f}")

print(f"\n📊 按选科组合的推荐统计:")
combo_stats = out.groupby('combo').agg({
    'est_score': ['mean', 'std', 'count'],
    '最低分': 'mean'
}).round(3)
combo_stats.columns = ['平均分数', '分数标准差', '推荐数量', '平均最低分']
print(combo_stats)

print(f"\n🎓 热门推荐专业 (前10):")
top_majors = out['专业名称'].value_counts().head(10)
print(top_majors)

# ---------- 12. 保存分析报告 ----------
analysis_report = DATA_DIR / "recommendation_analysis_report_sklearn.txt"
with open(analysis_report, 'w', encoding='utf-8') as f:
    f.write("=== 推荐系统分析报告 (scikit-learn SVD) ===\n\n")
    f.write(f"用户数量: {len(user_ids)}\n")
    f.write(f"专业数量: {len(item_ids)}\n")
    f.write(f"总推荐数量: {len(out)}\n")
    f.write(f"平均推荐分数: {out['est_score'].mean():.3f}\n")
    f.write(f"SVD成分数: {n_components}\n")
    f.write(f"解释方差比: {svd.explained_variance_ratio_.sum():.3f}\n")
    f.write(f"\n选科组合统计:\n")
    f.write(combo_stats.to_string())
    f.write(f"\n\n热门推荐专业:\n")
    f.write(top_majors.to_string())

print(f"✅ 分析报告已保存: {analysis_report}")
