# -*- coding: utf-8 -*-
"""
File: 008_可解释性分析_自动GPU_CPU.py
Purpose: PyTorch可解释性与SHAP分析（含自动设备）
Source notebook: 高考数据处理大修版本-checkpoint.ipynb
Execution order: keep numeric order 001 -> 010
"""

# ===== From Notebook CELL 75 =====
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
# 统一的工程/数据目录，避免硬编码绝对路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from gaokao_recommender.paths import *

# 导入统一的设备检测模块
from gaokao_recommender.device_utils import get_device, optimize_batch_size

DATA_DIR = DATA_RAW_DIR
OUTPUT_DIR = DATA_PROCESSED_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("使用 PyTorch 进行可解释性分析...")

# 自动选择计算设备：CUDA > MPS (Apple Silicon) > CPU
device, device_type = get_device(verbose=True)

# 优化 batch_size
BATCH_SIZE = optimize_batch_size(device, default_batch_size=64, verbose=True)

# ---------- 0. 路径和数据加载 ----------
BASE_DIR = PROJECT_ROOT
DATA_DIR = DATA_RAW_DIR

# 加载候选数据
try:
    cand = pd.read_csv(OUTPUT_DIR / "recommendation_candidates.csv", encoding="utf-8-sig")
    print("从文件加载候选数据")
except:
    print("❌ 请先运行推荐系统代码生成候选数据")
    exit()

print(f"候选数据形状: {cand.shape}")

# ---------- 1. 数据准备 ----------
FEATURES = ["mbti_w", "difficulty"]
X = cand[FEATURES].values.astype(np.float32)
y = cand["rating"].values.astype(np.float32)

print(f"特征数据形状: {X.shape}")

# 数据标准化
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_normalized = (X - X_mean) / X_std

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.2, random_state=42
)

print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

# ---------- 2. PyTorch 神经网络模型 ----------
class InterpretableNet(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 32]):
        super(InterpretableNet, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 创建模型
input_size = X_train.shape[1]
model = InterpretableNet(input_size).to(device)
print(f"模型结构:\n{model}")

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 转换为 PyTorch 张量
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

# 数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=(device_type == "cuda"))

# ---------- 3. 训练模型 ----------
def train_model(model, train_loader, epochs=100):
    model.train()
    train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=(device_type == "cuda"))
            batch_y = batch_y.to(device, non_blocking=(device_type == "cuda"))
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

    return train_losses

print("开始训练神经网络...")
train_losses = train_model(model, train_loader)

# ---------- 4. 模型评估 ----------
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor.to(device, non_blocking=(device_type == "cuda")))
    y_pred = y_pred_tensor.squeeze().cpu().numpy()

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 神经网络模型评估:")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# ---------- 5. PyTorch 可解释性分析 ----------
print("\n🔍 开始可解释性分析...")

# 5.1 梯度分析（类似 SHAP 的梯度解释）
def compute_feature_importance(model, X_tensor, feature_names):
    """使用梯度计算特征重要性"""
    model.eval()
    X_local = X_tensor.to(device, non_blocking=(device_type == "cuda")).clone().detach()
    X_local.requires_grad_(True)

    # 前向传播
    outputs = model(X_local)

    # 计算梯度
    gradients = []
    for i in range(outputs.shape[0]):
        model.zero_grad(set_to_none=True)
        if X_local.grad is not None:
            X_local.grad.zero_()
        outputs[i].backward(retain_graph=True)
        grad = X_local.grad[i].abs().detach().cpu().numpy()
        gradients.append(grad)

    gradients = np.array(gradients)
    importance_scores = gradients.mean(axis=0)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'gradient_importance': importance_scores
    }).sort_values('gradient_importance', ascending=False)

    return importance_df, gradients

# 计算梯度重要性
gradient_importance, all_gradients = compute_feature_importance(
    model, X_test_tensor[:1000], FEATURES  # 使用部分数据加速计算
)

print(f"\n🎯 梯度特征重要性:")
print(gradient_importance)

# 5.2 部分依赖分析（PDP）
def partial_dependence_analysis(model, feature_index, feature_values, X_base, feature_names):
    """计算部分依赖图"""
    pdp_values = []
    X_base_device = X_base.to(device, non_blocking=(device_type == "cuda"))

    for value in feature_values:
        X_temp = X_base_device.clone()
        X_temp[:, feature_index] = value
        with torch.no_grad():
            predictions = model(X_temp)
        pdp_values.append(predictions.mean().item())

    return pdp_values

# 为每个特征计算PDP
feature_values_dict = {}
X_test_device = X_test_tensor.to(device, non_blocking=(device_type == "cuda"))
for i, feature in enumerate(FEATURES):
    feature_min = X_test_device[:, i].min().item()
    feature_max = X_test_device[:, i].max().item()
    feature_values = torch.linspace(feature_min, feature_max, 50, device=device)
    feature_values_dict[feature] = feature_values

pdp_results = {}
for i, feature in enumerate(FEATURES):
    pdp_results[feature] = partial_dependence_analysis(
        model, i, feature_values_dict[feature], X_test_tensor, FEATURES
    )

# 5.3 个案分析
def analyze_individual_prediction(model, sample_index, X_tensor, feature_names):
    """分析单个预测"""
    model.eval()
    X_sample = X_tensor[sample_index:sample_index+1].to(device, non_blocking=(device_type == "cuda")).clone().detach()
    X_sample.requires_grad_(True)

    # 预测
    with torch.no_grad():
        prediction = model(X_sample).item()

    # 梯度
    model.zero_grad(set_to_none=True)
    output = model(X_sample)
    output.backward()
    gradients = X_sample.grad[0].abs().detach().cpu().numpy()

    print(f"\n样本 {sample_index} 分析:")
    print(f"预测评分: {prediction:.3f}")
    print(f"真实评分: {y_test[sample_index]:.3f}")
    print("特征贡献:")
    for i, feature in enumerate(feature_names):
        print(f"  {feature}: {gradients[i]:.4f}")

    return gradients

# ---------- 6. 可视化分析 ----------
plt.figure(figsize=(18, 12))

# 6.1 训练损失曲线
plt.subplot(2, 3, 1)
plt.plot(train_losses, color='#FF6B6B', linewidth=2)
plt.title('训练损失曲线', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

# 6.2 梯度特征重要性
plt.subplot(2, 3, 2)
colors = ['#4ECDC4', '#FF6B6B']
bars = plt.bar(gradient_importance['feature'],
               gradient_importance['gradient_importance'],
               color=colors)
plt.title('梯度特征重要性', fontsize=14, fontweight='bold')
plt.ylabel('平均梯度绝对值')
plt.xticks(rotation=45)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom')

# 6.3 部分依赖图 - MBTI匹配度
plt.subplot(2, 3, 3)
# 反标准化到原始尺度
mbti_original = feature_values_dict['mbti_w'].detach().cpu().numpy() * X_std[0] + X_mean[0]
plt.plot(mbti_original, pdp_results['mbti_w'], linewidth=3, color='#FF6B6B')
plt.xlabel('MBTI 匹配度 (原始尺度)')
plt.ylabel('平均预测评分')
plt.title('MBTI匹配度部分依赖图', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# 6.4 部分依赖图 - 难度
plt.subplot(2, 3, 4)
difficulty_original = feature_values_dict['difficulty'].detach().cpu().numpy() * X_std[1] + X_mean[1]
plt.plot(difficulty_original, pdp_results['difficulty'], linewidth=3, color='#4ECDC4')
plt.xlabel('难度 (原始尺度)')
plt.ylabel('平均预测评分')
plt.title('难度部分依赖图', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# 6.5 预测 vs 真实值
plt.subplot(2, 3, 5)
plt.scatter(y_test, y_pred, alpha=0.6, color='#96CEB4')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2)
plt.xlabel('真实评分')
plt.ylabel('预测评分')
plt.title('预测 vs 真实值', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# 6.6 残差分析
plt.subplot(2, 3, 6)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6, color='#45B7D1')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('预测评分')
plt.ylabel('残差')
plt.title('残差分析', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURE_DIR / "pytorch_interpretability.png", dpi=300, bbox_inches='tight')
plt.close()

# ---------- 7. 详细分析 ----------
print(f"\n🔍 详细分析结果:")

# 7.1 个案分析示例
print(f"\n🎯 个案分析 (前3个样本):")
for i in range(3):
    gradients = analyze_individual_prediction(model, i, X_test_tensor, FEATURES)

# 7.2 特征交互分析
print(f"\n📊 特征交互分析:")
# 计算特征相关性
feature_corr = np.corrcoef(X.T)
corr_df = pd.DataFrame(feature_corr, index=FEATURES, columns=FEATURES)
print("特征相关性矩阵:")
print(corr_df.round(3))

# 7.3 预测区间分析
print(f"\n📈 预测区间统计:")
prediction_intervals = {
    '1-2分': ((y_pred >= 1) & (y_pred < 2)).sum(),
    '2-3分': ((y_pred >= 2) & (y_pred < 3)).sum(),
    '3-4分': ((y_pred >= 3) & (y_pred < 4)).sum(),
    '4-5分': ((y_pred >= 4) & (y_pred <= 5)).sum()
}

for interval, count in prediction_intervals.items():
    percentage = count / len(y_pred) * 100
    print(f"  {interval}: {count} 个预测 ({percentage:.1f}%)")

# ---------- 8. 保存分析结果 ----------
analysis_results = PROJECT_ROOT / "reports" / "pytorch_interpretability_report.txt"
with open(analysis_results, 'w', encoding='utf-8') as f:
    f.write("=== PyTorch 可解释性分析报告 ===\n\n")
    f.write(f"数据规模: {cand.shape}\n")
    f.write(f"特征: {FEATURES}\n")
    f.write(f"模型性能 - MAE: {mae:.4f}, R²: {r2:.4f}\n\n")

    f.write("梯度特征重要性:\n")
    f.write(gradient_importance.to_string())

    f.write("\n\n特征相关性矩阵:\n")
    f.write(corr_df.to_string())

    f.write("\n\n预测区间统计:\n")
    for interval, count in prediction_intervals.items():
        percentage = count / len(y_pred) * 100
        f.write(f"{interval}: {count} ({percentage:.1f}%)\n")

# 保存模型
torch.save(model.state_dict(), MODEL_INTERPRETABILITY_DIR / "interpretable_model.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'X_mean': X_mean,
    'X_std': X_std,
    'feature_names': FEATURES
}, MODEL_INTERPRETABILITY_DIR / "interpretable_model_complete.pth")

print(f"\n✅ PyTorch 可解释性分析完成！")
print(f"📊 可视化已保存: {DATA_DIR / 'pytorch_interpretability.png'}")
print(f"📝 分析报告已保存: {analysis_results}")
print(f"🤖 模型已保存: {DATA_DIR / 'interpretable_model_complete.pth'}")


# ===== From Notebook CELL 76 =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
# 统一的工程/数据目录，避免硬编码绝对路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from gaokao_recommender.paths import *

DATA_DIR = DATA_RAW_DIR
OUTPUT_DIR = DATA_PROCESSED_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("开始 SHAP 可解释性分析...")

# ---------- 0. 路径和数据加载 ----------
BASE_DIR = PROJECT_ROOT
DATA_DIR = DATA_RAW_DIR

# 如果 cand 数据不存在，重新构建
try:
    cand = pd.read_csv(OUTPUT_DIR / "recommendation_candidates.csv", encoding="utf-8-sig")
    print("从文件加载候选数据")
except:
    print("重新构建候选数据...")
    # 重新构建 cand 数据（基于之前的推荐系统代码）
    df_combo = pd.read_csv(OUTPUT_DIR / "subject_combo_to_mbti_clean.csv", encoding="utf-8-sig")
    df_combo = df_combo.rename(columns={"subject_combo": "combo", "count": "count"})

    df_items = pd.read_csv(OUTPUT_DIR / "2023上海专业分数线_with_PredictedMBTI.csv", encoding="utf-8-sig")
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

    cand.to_csv(OUTPUT_DIR / "recommendation_candidates.csv", index=False, encoding="utf-8-sig")
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
    plt.savefig(FIGURE_DIR / "shap_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

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
    plt.savefig(FIGURE_DIR / "feature_analysis_alternative.png", dpi=300)
    plt.close()

# ---------- 7. 保存分析结果 ----------
analysis_results = PROJECT_ROOT / "reports" / "shap_analysis_results.txt"
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


