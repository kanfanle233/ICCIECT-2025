# -*- coding: utf-8 -*-
"""
File: 009_完整推荐流程与可视化输出.py
Purpose: 完整推荐流水线与用户摘要可视化
Source notebook: 高考数据处理大修版本-checkpoint.ipynb
Execution order: keep numeric order 001 -> 010
"""

# ===== From Notebook CELL 77 =====
import pandas as pd
import numpy as np

# 假设 cand 已经在前面构造好了，并包含列：mbti_w, difficulty, rating
# 例如：
# cand = pd.read_csv("combo_based_recommendations_input.csv")

from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt

# ---------- 1. 特征和目标 ----------
# 统一的工程/数据目录，避免硬编码绝对路径
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "志愿填报辅助系统" / "上海高考录取数据17-23年"
OUTPUT_DIR = DATA_DIR / "用到的"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURES = ["mbti_w", "difficulty"]  # 如果后面补上 gap，再加进来
X = cand[FEATURES]
y = cand["rating"]

# ---------- 2. 随机森林模型训练 ----------
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X, y)

# ---------- 3. SHAP 分析 ----------
# Create a TreeExplainer and compute SHAP values
explainer   = shap.Explainer(model, X)
shap_values = explainer(X)






# ===== From Notebook CELL 78 =====
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ===（假设 cand 已经定义好）===
# cand = pd.read_csv("combo_based_recommendations_input.csv")
# 它包含列：mbti_w, difficulty, rating

FEATURES = ["mbti_w", "difficulty"]
X = cand[FEATURES]
y = cand["rating"]

# 1. 训练随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X, y)

# 2. 用统一 API 创建 Explainer 并计算 SHAP 值
explainer = shap.Explainer(model, X)   # 默认会选择 TreeExplainer
shap_values = explainer(X)

# 3. 检查基础形状
print("base_values shape:", shap_values.base_values.shape)  # (n_samples,)
print("values shape      :", shap_values.values.shape)      # (n_samples, 2)
print("feature names     :", shap_values.feature_names)     # ['mbti_w', 'difficulty']

# 4. 全局 summary plot
plt.figure(figsize=(8,6))
shap.summary_plot(shap_values, X, feature_names=FEATURES)

# 5. 局部解释：Waterfall Plot，示例第 0 个样本
i = 0
base_i      = shap_values.base_values[i]   # 标量
shap_vals_i = shap_values.values[i]        # (2,)

print(f"Sample {i} base value:", base_i)
print(f"Sample {i} shap values:", shap_vals_i)

plt.figure(figsize=(6,4))
shap.plots.waterfall(
    shap.Explanation(
        values=shap_vals_i,
        base_values=base_i,
        data=X.iloc[i].values,    # 如果 X 是 DataFrame
        feature_names=FEATURES
    )
)
plt.title(f"Waterfall Plot for Sample {i}")
plt.tight_layout()
plt.show()

# 6. （可选） 使用 legacy waterfall
plt.figure(figsize=(6,4))
shap.plots._waterfall.waterfall_legacy(
    base_i,
    shap_vals_i,
    FEATURES
)
plt.title(f"Legacy Waterfall for Sample {i}")
plt.tight_layout()
plt.show()

# 7. （可选） Partial Dependence / 依赖图
shap.dependence_plot(
    "difficulty", 
    shap_values.values,
    X,
    feature_names=FEATURES
)




# ===== From Notebook CELL 79 =====
import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# === 假设 cand 已经准备好，并且包含列：mbti_w, difficulty, rating ===
# cand = pd.read_csv("combo_based_recommendations_input.csv")

# 特征与目标
FEATURES = ["mbti_w", "difficulty"]
X = cand[FEATURES]
y = cand["rating"]

# 1. 训练随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X, y)

# 2. 用统一 API 创建 Explainer 并计算 SHAP 值
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# 3. 确认要保存的目录是否存在，不存在则创建
save_dir = OUTPUT_DIR
os.makedirs(save_dir, exist_ok=True)

# === 4. 绘制并保存全局 Summary Plot ===
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, X, feature_names=FEATURES, show=False)  # show=False 不直接弹窗
summary_path = os.path.join(save_dir, "shap_summary_plot.png")
plt.savefig(summary_path, dpi=300, bbox_inches="tight")
plt.close()  # 关闭当前 Figure

print(f"✅ 全局 Summary Plot 已保存到：\n    {summary_path}")

# === 5. 绘制并保存第 0 个样本的局部 Waterfall Plot ===
i = 0
base_i = shap_values.base_values[i]    # 第 0 个样本的 base value（标量）
shap_vals_i = shap_values.values[i]    # 第 0 个样本的 SHAP 向量（长度 = len(FEATURES)）

# 使用新版 API 的 waterfall
plt.figure(figsize=(6, 4))
shap.plots.waterfall(
    shap.Explanation(
        values=shap_vals_i,
        base_values=base_i,
        data=X.iloc[i].values,    # 原始特征取值
        feature_names=FEATURES
    ),
    show=False
)
waterfall_path = os.path.join(save_dir, "shap_waterfall_sample0.png")
plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ 第 0 个样本的 Waterfall Plot 已保存到：\n    {waterfall_path}")

# === 6.（可选） 保存 Legacy Waterfall Plot ===
plt.figure(figsize=(6, 4))
shap.plots._waterfall.waterfall_legacy(
    base_i,
    shap_vals_i,
    FEATURES,
    show=False
)
legacy_path = os.path.join(save_dir, "shap_waterfall_legacy_sample0.png")
plt.savefig(legacy_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ 旧版 Legacy Waterfall Plot 已保存到：\n    {legacy_path}")

# === 7.（可选） 保存 Partial Dependence Plot（以 ’difficulty‘ 为例）===
plt.figure(figsize=(6, 4))
shap.dependence_plot(
    "difficulty",
    shap_values.values,
    X,
    feature_names=FEATURES,
    show=False
)
dependence_path = os.path.join(save_dir, "shap_dependence_difficulty.png")
plt.savefig(dependence_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ Partial Dependence Plot 已保存到：\n    {dependence_path}")




# ===== From Notebook CELL 80 =====
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

# 设置中文字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

print("开始完整的推荐系统和可视化流程...")

# ---------- 1. 路径设置 ----------
BASE_DIR = PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "志愿填报辅助系统" / "上海高考录取数据17-23年"

# ---------- 2. 数据加载 ----------
try:
    # 加载选科组合数据
    df_combo = pd.read_csv(DATA_DIR / "subject_combo_to_mbti_clean.csv", encoding="utf-8-sig")
    df_combo = df_combo.rename(columns={"subject_combo": "combo", "count": "count"})

    # 加载专业MBTI数据
    df_items = pd.read_csv(DATA_DIR / "2023上海专业分数线_with_PredictedMBTI.csv", encoding="utf-8-sig")
    df_items["school_major"] = df_items["院校名称"] + "_" + df_items["专业名称"]

    print(f"选科组合数据形状: {df_combo.shape}")
    print(f"专业数据形状: {df_items.shape}")

except FileNotFoundError as e:
    print(f"❌ 数据文件不存在: {e}")
    exit()

# ---------- 3. 学生画像 ----------
df_users = df_combo[["combo", "mbti", "count"]].copy()
df_users["uid"] = "u_" + df_users.index.astype(str)

print(f"用户数据形状: {df_users.shape}")

# ---------- 4. MBTI匹配度函数 ----------
def mbti_weight(stu_mbti, maj_mbti):
    if stu_mbti == maj_mbti:
        return 1.0
    if stu_mbti[:2] == maj_mbti[:2]:
        return 0.5
    return 0.0

# ---------- 5. 生成伪评分表 ----------
print("生成候选评分矩阵...")

batch_difficulty = {
    "专科批": 1, "本科批": 3, "本科提前批": 4,
    "高职提前批": 2, "提前批": 4, "艺术类本科批": 3, "体育类本科批": 3
}
max_difficulty = float(max(batch_difficulty.values()))

# 向量化构造候选集（保持原有字段和评分公式）
users_small = df_users[["uid", "combo", "mbti"]].copy()
items_small = df_items[["school_major", "院校名称", "专业名称", "Predicted_MBTI", "最低分", "批次"]].copy()

cand = users_small.assign(_k=1).merge(items_small.assign(_k=1), on="_k").drop(columns="_k")

user_mbti = cand["mbti"].fillna("").astype(str)
item_mbti = cand["Predicted_MBTI"].fillna("").astype(str)

exact_match = user_mbti.eq(item_mbti)
prefix_match = user_mbti.str[:2].eq(item_mbti.str[:2])

cand["mbti_w"] = np.select([exact_match, prefix_match], [1.0, 0.5], default=0.0).astype(np.float32)
cand["difficulty"] = (
    cand["批次"].map(batch_difficulty).fillna(3).astype(np.float32) / max_difficulty
)
cand["rating"] = (cand["mbti_w"] * (1.0 - cand["difficulty"]) * 4.0 + 1.0).astype(np.float32)

cand = cand[[
    "uid", "combo", "mbti", "school_major", "mbti_w", "difficulty", "rating",
    "院校名称", "专业名称", "Predicted_MBTI", "最低分"
]]

print(f"候选数据形状: {cand.shape}")

# ---------- 6. 矩阵分解推荐 ----------
print("开始矩阵分解推荐...")

# 创建用户-项目评分矩阵
user_ids = df_users["uid"].tolist()
item_ids = df_items["school_major"].tolist()

item_difficulty = (
    df_items["批次"].map(batch_difficulty).fillna(3).astype(np.float32).to_numpy() / max_difficulty
)
user_mbti_arr = df_users["mbti"].fillna("").astype(str).to_numpy()
item_mbti_arr = df_items["Predicted_MBTI"].fillna("").astype(str).to_numpy()

user_prefix = np.array([s[:2] for s in user_mbti_arr], dtype=object)
item_prefix = np.array([s[:2] for s in item_mbti_arr], dtype=object)

exact_match_m = user_mbti_arr[:, None] == item_mbti_arr[None, :]
prefix_match_m = user_prefix[:, None] == item_prefix[None, :]
mbti_matrix = np.where(exact_match_m, 1.0, np.where(prefix_match_m, 0.5, 0.0)).astype(np.float32)

rating_matrix = (mbti_matrix * (1.0 - item_difficulty[None, :]) * 4.0 + 1.0).astype(np.float32)

# 使用TruncatedSVD
n_components = min(50, min(rating_matrix.shape) - 1)
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_factors = svd.fit_transform(rating_matrix)
item_factors = svd.components_.T

# 重建评分矩阵
reconstructed_ratings = user_factors @ item_factors.T

print(f"SVD分解完成: {n_components} 个成分")

# ---------- 7. 生成推荐结果 ----------
def recommend(user_index, N=10):
    user_scores = reconstructed_ratings[user_index]
    top_indices = np.argsort(user_scores)[::-1][:N]
    return [(item_ids[j], user_scores[j]) for j in top_indices]

TOP_N = 10
item_ids_arr = np.array(item_ids, dtype=object)

raw_top_idx = np.argpartition(-reconstructed_ratings, kth=TOP_N - 1, axis=1)[:, :TOP_N]
raw_top_scores = np.take_along_axis(reconstructed_ratings, raw_top_idx, axis=1)
order = np.argsort(-raw_top_scores, axis=1)
top_idx = np.take_along_axis(raw_top_idx, order, axis=1)
top_scores = np.take_along_axis(raw_top_scores, order, axis=1)

out = pd.DataFrame({
    "uid": np.repeat(np.array(user_ids, dtype=object), TOP_N),
    "rank": np.tile(np.arange(1, TOP_N + 1), len(user_ids)),
    "school_major": item_ids_arr[top_idx.reshape(-1)],
    "est_score": top_scores.reshape(-1)
})

# 添加用户信息和专业信息
out = out.merge(df_users[["uid", "combo", "mbti"]], on="uid")
out = out.merge(df_items[["school_major", "院校名称", "专业名称", "最低分", "Predicted_MBTI"]], on="school_major")

# 重新排列列顺序
out = out[["uid", "combo", "mbti", "rank", "院校名称", "专业名称", "最低分", "Predicted_MBTI", "est_score"]]

# 保存推荐结果
recommendations_file = DATA_DIR / "combo_based_recommendations.csv"
out.to_csv(recommendations_file, index=False, encoding="utf-8-sig")
print(f"✅ 推荐结果已保存: {recommendations_file}")

# ---------- 8. 单个用户推荐可视化 ----------
print(f"\n开始可视化推荐结果...")

# 选择第一个用户进行可视化
available_users = out['uid'].unique()
if len(available_users) > 0:
    uid = available_users[0]
    print(f"选择用户: {uid}")
else:
    print("❌ 没有可用的用户数据")
    exit()

# 获取该用户的Top10推荐
user_recommendations = out[out['uid'] == uid].head(10)

if len(user_recommendations) == 0:
    print(f"❌ 用户 {uid} 没有推荐结果")
    exit()

print(f"用户 {uid} 的推荐结果:")
print(user_recommendations[['院校名称', '专业名称', 'est_score', 'mbti', 'Predicted_MBTI']])

# 准备可视化数据
user_recommendations['display_name'] = user_recommendations['院校名称'] + ' - ' + user_recommendations['专业名称']
user_recommendations = user_recommendations.sort_values('est_score', ascending=True)

# 可视化绘图
plt.figure(figsize=(12, 8))

# 创建颜色映射
colors = plt.cm.viridis((user_recommendations['est_score'] - user_recommendations['est_score'].min()) /
                        (user_recommendations['est_score'].max() - user_recommendations['est_score'].min()))

# 水平条形图
bars = plt.barh(user_recommendations['display_name'],
                user_recommendations['est_score'],
                color=colors,
                alpha=0.8,
                height=0.7)

# 美化图表
plt.xlabel('推荐分数', fontsize=12, fontweight='bold')
plt.ylabel('院校-专业', fontsize=12, fontweight='bold')

# 获取用户信息用于标题
user_info = user_recommendations[['combo', 'mbti']].iloc[0]
plt.title(f'{uid}\n选科组合: {user_info["combo"]} | MBTI: {user_info["mbti"]}\nTop-10 推荐志愿',
          fontsize=14, fontweight='bold', pad=20)

# 添加分数标签
for i, (bar, score) in enumerate(zip(bars, user_recommendations['est_score'])):
    width = bar.get_width()
    plt.text(width + 0.05, bar.get_y() + bar.get_height()/2,
             f'{score:.3f}',
             ha='left', va='center', fontsize=10, fontweight='bold')

# 设置x轴范围
x_max = user_recommendations['est_score'].max()
plt.xlim(0, x_max + 0.5)

# 添加网格
plt.grid(True, axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()

# 保存图像
save_path = DATA_DIR / f"{uid}_top10_recommendations.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ 推荐结果可视化已保存到: {save_path}")

plt.show()

# ---------- 9. MBTI匹配分析 ----------
print(f"\n🔍 MBTI匹配分析:")
mbti_matches = user_recommendations['Predicted_MBTI'].value_counts()
print("推荐专业的MBTI类型分布:")
for mbti, count in mbti_matches.items():
    match_type = "完全匹配" if mbti == user_info['mbti'] else "部分匹配" if mbti[:2] == user_info['mbti'][:2] else "其他"
    print(f"  {mbti}: {count} 个专业 ({match_type})")

# ---------- 10. 所有用户摘要可视化 ----------
print(f"\n📊 创建所有用户的推荐摘要...")

# 计算每个用户的平均推荐分数
user_stats = out.groupby(['uid', 'combo', 'mbti']).agg({
    'est_score': ['mean', 'std', 'count'],
    '最低分': 'mean'
}).round(3)

user_stats.columns = ['平均分数', '分数标准差', '推荐数量', '平均最低分']
user_stats = user_stats.reset_index()

# 可视化所有用户的平均分数
plt.figure(figsize=(14, 8))

# 按MBTI分组着色
mbti_colors = {
    'ISTJ': '#FF6B6B', 'ESTJ': '#4ECDC4', 'INTP': '#45B7D1',
    'INTJ': '#96CEB4', 'INFJ': '#FECA57', 'ESFP': '#FF9FF3',
    'ENFJ': '#54A0FF', 'ESFJ': '#5F27CD', 'ISFJ': '#00D2D3',
    'ENTJ': '#FF9F43', 'ENFP': '#10AC84', 'ESTP': '#EE5A24'
}

user_stats['color'] = user_stats['mbti'].map(mbti_colors)
user_stats['color'] = user_stats['color'].fillna('#C8D6E5')

# 创建散点图
plt.scatter(user_stats['平均分数'], user_stats['平均最低分'],
           c=user_stats['color'], s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

plt.xlabel('平均推荐分数', fontsize=12, fontweight='bold')
plt.ylabel('平均录取最低分', fontsize=12, fontweight='bold')
plt.title('所有用户推荐结果摘要\n(点的大小表示推荐数量)', fontsize=14, fontweight='bold')

# 添加图例
existing_mbti = user_stats['mbti'].unique()
legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=mbti_colors.get(mbti, '#C8D6E5'),
                             markersize=8, label=mbti)
                  for mbti in existing_mbti if mbti in mbti_colors]
plt.legend(handles=legend_elements, title='MBTI类型', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存摘要图
summary_path = DATA_DIR / "all_users_recommendation_summary.png"
plt.savefig(summary_path, dpi=300, bbox_inches='tight')
print(f"✅ 所有用户推荐摘要已保存到: {summary_path}")

plt.show()

print(f"\n🎉 完整流程完成!")
print(f"📁 推荐结果文件: {recommendations_file}")
print(f"📊 单个用户图表: {save_path}")
print(f"📈 摘要图表: {summary_path}")


# ===== From Notebook CELL 81 =====
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体（防止中文乱码）
plt.rcParams['font.family'] = 'SimHei'  # Windows 系统默认黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ---------- 1. 路径设置 ----------
BASE_DIR = PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "志愿填报辅助系统" / "上海高考录取数据17-23年"

# ---------- 2. 加载真实的推荐结果 ----------
try:
    # 尝试加载我们之前生成的推荐结果
    recommendations = pd.read_csv(DATA_DIR / "combo_based_recommendations.csv", encoding="utf-8-sig")
    print(f"成功加载推荐数据，形状: {recommendations.shape}")

    # 显示可用的用户ID
    available_users = recommendations['uid'].unique()
    print(f"可用的用户ID (前10个): {available_users[:10]}")

except FileNotFoundError:
    print("❌ 推荐结果文件不存在，请先运行推荐系统代码")
    exit()

# ---------- 3. 选择要可视化的用户 ----------
# 方法1：选择第一个用户
if len(available_users) > 0:
    uid = available_users[0]
    print(f"选择用户: {uid}")
else:
    print("❌ 没有可用的用户数据")
    exit()

# 方法2：或者选择特定的MBTI类型用户
# esfp_users = recommendations[recommendations['mbti'] == 'ESFP']['uid'].unique()
# if len(esfp_users) > 0:
#     uid = esfp_users[0]
#     print(f"选择ESFP用户: {uid}")

# ---------- 4. 获取该用户的Top10推荐 ----------
user_recommendations = recommendations[recommendations['uid'] == uid].head(10)

if len(user_recommendations) == 0:
    print(f"❌ 用户 {uid} 没有推荐结果")
    exit()

print(f"用户 {uid} 的推荐结果:")
print(user_recommendations[['院校名称', '专业名称', 'est_score', 'mbti', 'Predicted_MBTI']])

# ---------- 5. 准备可视化数据 ----------
# 创建显示名称（院校-专业）
user_recommendations['display_name'] = user_recommendations['院校名称'] + ' - ' + user_recommendations['专业名称']

# 按分数排序
user_recommendations = user_recommendations.sort_values('est_score', ascending=True)  # 为水平条形图排序

# ---------- 6. 可视化绘图 ----------
plt.figure(figsize=(12, 8))

# 创建颜色映射，根据分数从低到高渐变
colors = plt.cm.viridis((user_recommendations['est_score'] - user_recommendations['est_score'].min()) /
                        (user_recommendations['est_score'].max() - user_recommendations['est_score'].min()))

# 水平条形图
bars = plt.barh(user_recommendations['display_name'],
                user_recommendations['est_score'],
                color=colors,
                alpha=0.8,
                height=0.7)

# 美化图表
plt.xlabel('推荐分数', fontsize=12, fontweight='bold')
plt.ylabel('院校-专业', fontsize=12, fontweight='bold')

# 获取用户信息用于标题
user_info = user_recommendations[['combo', 'mbti']].iloc[0]
plt.title(f'{uid}\n选科组合: {user_info["combo"]} | MBTI: {user_info["mbti"]}\nTop-10 推荐志愿',
          fontsize=14, fontweight='bold', pad=20)

# 添加分数标签
for i, (bar, score) in enumerate(zip(bars, user_recommendations['est_score'])):
    width = bar.get_width()
    plt.text(width + 0.05, bar.get_y() + bar.get_height()/2,
             f'{score:.3f}',
             ha='left', va='center', fontsize=10, fontweight='bold')

# 设置x轴范围，为分数标签留出空间
x_max = user_recommendations['est_score'].max()
plt.xlim(0, x_max + 0.5)

# 添加网格
plt.grid(True, axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()

# ---------- 7. 保存图像 ----------
save_path = DATA_DIR / f"{uid}_top10_recommendations.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ 推荐结果可视化已保存到: {save_path}")

plt.show()

# ---------- 8. 额外分析：显示MBTI匹配情况 ----------
print(f"\n🔍 MBTI匹配分析:")
mbti_matches = user_recommendations['Predicted_MBTI'].value_counts()
print("推荐专业的MBTI类型分布:")
for mbti, count in mbti_matches.items():
    match_type = "完全匹配" if mbti == user_info['mbti'] else "部分匹配" if mbti[:2] == user_info['mbti'][:2] else "其他"
    print(f"  {mbti}: {count} 个专业 ({match_type})")

# ---------- 9. 创建所有用户的摘要可视化 ----------
print(f"\n📊 创建所有用户的推荐摘要...")

# 计算每个用户的平均推荐分数
user_stats = recommendations.groupby(['uid', 'combo', 'mbti']).agg({
    'est_score': ['mean', 'std', 'count'],
    '最低分': 'mean'
}).round(3)

user_stats.columns = ['平均分数', '分数标准差', '推荐数量', '平均最低分']
user_stats = user_stats.reset_index()

# 可视化所有用户的平均分数
plt.figure(figsize=(14, 8))

# 按MBTI分组着色
mbti_colors = {
    'ISTJ': '#FF6B6B', 'ESTJ': '#4ECDC4', 'INTP': '#45B7D1',
    'INTJ': '#96CEB4', 'INFJ': '#FECA57', 'ESFP': '#FF9FF3',
    'ENFJ': '#54A0FF', 'ESFJ': '#5F27CD', 'ISFJ': '#00D2D3',
    'ENTJ': '#FF9F43', 'ENFP': '#10AC84', 'ESTP': '#EE5A24'
}

# 为每个MBTI类型分配颜色
user_stats['color'] = user_stats['mbti'].map(mbti_colors)
# 对于没有指定颜色的MBTI，使用默认颜色
user_stats['color'] = user_stats['color'].fillna('#C8D6E5')

# 创建散点图
scatter = plt.scatter(user_stats['平均分数'], user_stats['平均最低分'],
                     c=user_stats['color'], s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

plt.xlabel('平均推荐分数', fontsize=12, fontweight='bold')
plt.ylabel('平均录取最低分', fontsize=12, fontweight='bold')
plt.title('所有用户推荐结果摘要\n(点的大小表示推荐数量)', fontsize=14, fontweight='bold')

# 添加图例（只显示存在的MBTI类型）
existing_mbti = user_stats['mbti'].unique()
legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=mbti_colors.get(mbti, '#C8D6E5'),
                             markersize=8, label=mbti)
                  for mbti in existing_mbti if mbti in mbti_colors]
plt.legend(handles=legend_elements, title='MBTI类型', bbox_to_anchor=(1.05, 1), loc='upper left')

# 添加数据点标签（只标注部分点避免拥挤）
for i, row in user_stats.iterrows():
    if i % 3 == 0:  # 每3个点标注一个
        plt.annotate(row['combo'], (row['平均分数'], row['平均最低分']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存摘要图
summary_path = DATA_DIR / "all_users_recommendation_summary.png"
plt.savefig(summary_path, dpi=300, bbox_inches='tight')
print(f"✅ 所有用户推荐摘要已保存到: {summary_path}")

plt.show()

print(f"\n🎉 可视化完成!")
print(f"📁 单个用户图表: {save_path}")
print(f"📊 摘要图表: {summary_path}")




