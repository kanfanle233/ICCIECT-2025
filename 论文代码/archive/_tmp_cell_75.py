"""
高考志愿填报辅助系统 - PyTorch可解释性神经网络分析

使用 PyTorch 构建神经网络回归模型，通过梯度分析、部分依赖图（PDP）
等方法对模型进行可解释性分析，探索特征对推荐评分的影响机制。
教学重点：深度学习模型的可解释性——梯度分析与部分依赖图。
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path

print("使用 PyTorch 进行可解释性分析...")

# 自动选择计算设备：优先 CUDA，不可用则回退 CPU
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
if USE_CUDA:
    torch.backends.cudnn.benchmark = True
    print(f"检测到 CUDA，可用设备: {torch.cuda.get_device_name(0)}")
else:
    print("未检测到可用 CUDA，使用 CPU 运行")

# ---------- 0. 路径和数据加载 ----------
BASE_DIR = Path("E:/代码/minicondapythonProject1")
DATA_DIR = BASE_DIR / "论文代码" / "志愿填报辅助系统" / "上海高考录取数据17-23年"

# 加载候选数据
try:
    cand = pd.read_csv(DATA_DIR / "recommendation_candidates.csv", encoding="utf-8-sig")
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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=USE_CUDA)

# ---------- 3. 训练模型 ----------
def train_model(model, train_loader, epochs=100):
    model.train()
    train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=USE_CUDA)
            batch_y = batch_y.to(device, non_blocking=USE_CUDA)
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
    y_pred_tensor = model(X_test_tensor.to(device, non_blocking=USE_CUDA))
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
    X_local = X_tensor.to(device, non_blocking=USE_CUDA).clone().detach()
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
    X_base_device = X_base.to(device, non_blocking=USE_CUDA)

    for value in feature_values:
        X_temp = X_base_device.clone()
        X_temp[:, feature_index] = value
        with torch.no_grad():
            predictions = model(X_temp)
        pdp_values.append(predictions.mean().item())

    return pdp_values

# 为每个特征计算PDP
feature_values_dict = {}
X_test_device = X_test_tensor.to(device, non_blocking=USE_CUDA)
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
    X_sample = X_tensor[sample_index:sample_index+1].to(device, non_blocking=USE_CUDA).clone().detach()
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
plt.savefig(DATA_DIR / "pytorch_interpretability.png", dpi=300, bbox_inches='tight')
plt.show()

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
analysis_results = DATA_DIR / "pytorch_interpretability_report.txt"
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
torch.save(model.state_dict(), DATA_DIR / "interpretable_model.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'X_mean': X_mean,
    'X_std': X_std,
    'feature_names': FEATURES
}, DATA_DIR / "interpretable_model_complete.pth")

print(f"\n✅ PyTorch 可解释性分析完成！")
print(f"📊 可视化已保存: {DATA_DIR / 'pytorch_interpretability.png'}")
print(f"📝 分析报告已保存: {analysis_results}")
print(f"🤖 模型已保存: {DATA_DIR / 'interpretable_model_complete.pth'}")
