"""
高考志愿填报辅助系统 - SHAP可解释性分析

使用 SHAP (SHapley Additive exPlanations) 对随机森林推荐模型进行
全局和局部可解释性分析，揭示 MBTI 匹配度和批次难度对推荐评分的影响。
教学重点：机器学习模型的可解释性方法——SHAP值分析。
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path

print("开始 SHAP 可解释性分析...")

# ---------- 0. 路径和数据加载 ----------
BASE_DIR = Path("E:/代码/minicondapythonProject1")
DATA_DIR = BASE_DIR / "论文代码" / "志愿填报辅助系统" / "上海高考录取数据17-23年"

# 如果 cand 数据不存在，重新构建
try:
    cand = pd.read_csv(DATA_DIR / "recommendation_candidates.csv", encoding="utf-8-sig")
    print("从文件加载候选数据")
except:
    print("重新构建候选数据...")
    # 重新构建 cand 数据（基于之前的推荐系统代码）
    df_combo = pd.read_csv(DATA_DIR / "subject_combo_to_mbti_clean.csv", encoding="utf-8-sig")
    df_combo = df_combo.rename(columns={"subject_combo": "combo", "count": "count"})

    df_items = pd.read_csv(DATA_DIR / "2023上海专业分数线_with_PredictedMBTI.csv", encoding="utf-8-sig")
    df_items["school_major"] = df_items["院校名称"] + "_" + df_items["专业名称"]

    df_users = df_combo[["combo", "mbti", "count"]].copy()
    df_users["uid"] = "u_" + df_users.index.astype(str)

    # 批次难度映射
    batch_difficulty = {
        "专科批": 1, "本科批": 3, "本科提前批": 4,
        "高职提前批": 2, "提前批": 4, "艺术类本科批": 3, "体育类本科批": 3
    }
    max_difficulty = float(max(batch_difficulty.values()))

    # 向量化构造候选数据（保持原有逻辑）
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

    cand.to_csv(DATA_DIR / "recommendation_candidates.csv", index=False, encoding="utf-8-sig")
    print("候选数据已保存")

print(f"候选数据形状: {cand.shape}")
print(f"列名: {list(cand.columns)}")

# ---------- 1. 特征和目标 ----------
FEATURES = ["mbti_w", "difficulty"]
X = cand[FEATURES]
y = cand["rating"]

print(f"特征数据形状: {X.shape}")
print(f"目标数据形状: {y.shape}")
print(f"特征统计:\n{X.describe()}")

# ---------- 2. 随机森林模型训练 ----------
from sklearn.model_selection import train_test_split

# 分割训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train, y_train)

# 模型评估
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 随机森林模型评估:")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# 特征重要性
feature_importance = pd.DataFrame({
    'feature': FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🎯 特征重要性:")
print(feature_importance)

# ---------- 3. SHAP 分析 ----------
try:
    import shap
    print("开始 SHAP 分析...")

    # 使用 TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    print(f"SHAP值计算完成，形状: {np.array(shap_values).shape}")

    # ---------- 4. SHAP 可视化 ----------
    plt.figure(figsize=(15, 12))

    # 4.1 SHAP 摘要图
    plt.subplot(2, 2, 1)
    shap.summary_plot(shap_values, X, show=False)
    plt.title('SHAP 特征重要性摘要', fontsize=14, fontweight='bold')

    # 4.2 SHAP 条形图（特征重要性）
    plt.subplot(2, 2, 2)
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title('SHAP 特征重要性（条形图）', fontsize=14, fontweight='bold')

    # 4.3 特征重要性对比
    plt.subplot(2, 2, 3)
    colors = ['#FF6B6B', '#4ECDC4']
    bars = plt.bar(feature_importance['feature'], feature_importance['importance'], color=colors)
    plt.title('随机森林特征重要性', fontsize=14, fontweight='bold')
    plt.ylabel('重要性分数')
    plt.xticks(rotation=45)

    # 在柱子上添加数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')

    # 4.4 特征关系散点图
    plt.subplot(2, 2, 4)
    plt.scatter(cand['mbti_w'], cand['rating'], alpha=0.6, c=cand['difficulty'], cmap='viridis')
    plt.colorbar(label='难度')
    plt.xlabel('MBTI 匹配度')
    plt.ylabel('评分')
    plt.title('MBTI匹配度 vs 评分（颜色表示难度）', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(DATA_DIR / "shap_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

    # ---------- 5. 详细分析 ----------
    print(f"\n🔍 SHAP 分析详情:")

    # 计算平均 |SHAP| 值
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        'feature': FEATURES,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)

    print("SHAP 特征重要性:")
    print(shap_importance)

    # 分析特征对评分的影响方向
    print(f"\n📈 特征影响方向:")
    for i, feature in enumerate(FEATURES):
        corr = np.corrcoef(X[feature], shap_values[:, i])[0, 1]
        print(f"  {feature}: 相关性 = {corr:.4f}")

        # 分析特征值范围对SHAP值的影响
        high_values = X[feature] > X[feature].median()
        low_values = X[feature] <= X[feature].median()

        mean_shap_high = shap_values[high_values, i].mean()
        mean_shap_low = shap_values[low_values, i].mean()

        print(f"    高值平均SHAP: {mean_shap_high:.4f}, 低值平均SHAP: {mean_shap_low:.4f}")

    # ---------- 6. 个案分析 ----------
    print(f"\n🎯 个案分析 (前5个样本):")
    sample_indices = range(5)
    for idx in sample_indices:
        print(f"\n样本 {idx}:")
        print(f"  真实评分: {y.iloc[idx]:.3f}")
        print(f"  预测评分: {model.predict(X.iloc[[idx]])[0]:.3f}")
        print(f"  MBTI匹配度: {X.iloc[idx]['mbti_w']:.3f}")
        print(f"  难度: {X.iloc[idx]['difficulty']:.3f}")
        print(f"  SHAP贡献: MBTI匹配度 = {shap_values[idx, 0]:.4f}, 难度 = {shap_values[idx, 1]:.4f}")
        print(f"  基准值: {explainer.expected_value:.4f}")

except ImportError:
    print("❌ SHAP 库未安装，使用替代可视化")
    print("请运行: pip install shap")

    # 替代可视化
    plt.figure(figsize=(12, 8))

    # 特征重要性
    plt.subplot(2, 2, 1)
    colors = ['#FF6B6B', '#4ECDC4']
    bars = plt.bar(feature_importance['feature'], feature_importance['importance'], color=colors)
    plt.title('随机森林特征重要性')
    plt.ylabel('重要性分数')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')

    # 特征与评分的关系
    plt.subplot(2, 2, 2)
    plt.scatter(cand['mbti_w'], cand['rating'], alpha=0.6)
    plt.xlabel('MBTI 匹配度')
    plt.ylabel('评分')
    plt.title('MBTI匹配度 vs 评分')

    plt.subplot(2, 2, 3)
    plt.scatter(cand['difficulty'], cand['rating'], alpha=0.6)
    plt.xlabel('难度')
    plt.ylabel('评分')
    plt.title('难度 vs 评分')

    # 残差分析
    plt.subplot(2, 2, 4)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差分析')

    plt.tight_layout()
    plt.savefig(DATA_DIR / "feature_analysis_alternative.png", dpi=300)
    plt.show()

# ---------- 7. 保存分析结果 ----------
analysis_results = DATA_DIR / "shap_analysis_results.txt"
with open(analysis_results, 'w', encoding='utf-8') as f:
    f.write("=== SHAP 可解释性分析报告 ===\n\n")
    f.write(f"数据规模: {cand.shape}\n")
    f.write(f"特征: {FEATURES}\n")
    f.write(f"模型性能 - MAE: {mae:.4f}, R²: {r2:.4f}\n\n")
    f.write("特征重要性:\n")
    f.write(feature_importance.to_string())

    if 'shap_importance' in locals():
        f.write("\n\nSHAP 特征重要性:\n")
        f.write(shap_importance.to_string())

print(f"\n✅ 分析完成！")
print(f"📊 可视化已保存: {DATA_DIR / 'shap_analysis.png'}")
print(f"📝 分析报告已保存: {analysis_results}")
