"""
高考志愿填报辅助系统 - PyTorch LSTM优化模型训练

使用 PyTorch 实现 LSTM 模型进行分数→百分位的回归预测，
包含自动GPU/CPU检测、早停机制、模型保存等功能。
教学重点：PyTorch LSTM 时序回归模型的工程化训练与评估。
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

print("开始构建优化模型...")

# 自动选择计算设备：优先 CUDA，不可用则回退 CPU
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
if USE_CUDA:
    torch.backends.cudnn.benchmark = True
    print(f"检测到 CUDA，可用设备: {torch.cuda.get_device_name(0)}")
else:
    print("未检测到可用 CUDA，使用 CPU 运行")

# 0. 路径
BASE_DIR = Path("E:/代码/minicondapythonProject1")
DATA_DIR = BASE_DIR / "论文代码" / "志愿填报辅助系统" / "上海高考录取数据17-23年"
PATH_RANK = DATA_DIR / "2023年考生高考成绩分布表_clean.csv"
PATH_MAJOR = DATA_DIR / "2023上海专业分数线_model_ready.csv"

# 1. 读取数据
df_rank = pd.read_csv(PATH_RANK, encoding="utf-8-sig")
df_major = pd.read_csv(PATH_MAJOR, encoding="utf-8-sig")

print(f"成绩分布数据形状: {df_rank.shape}")
print(f"专业数据形状: {df_major.shape}")

# 2. 确保 df_major 只有 2023 年的那一行
df_major_2023 = df_major.copy()

# 3. 把专业表的数值特征"广播"到所有分数行
for col in ["批次_code", "最低分", "最低位次", "平均分"]:
    if col in df_major_2023.columns:
        df_rank[col] = df_major_2023[col].iloc[0]
    else:
        print(f"警告: 列 '{col}' 不存在于专业数据中")

# 4. 构造衍生特征
total_n = df_rank["累计人数"].max()
df_rank["score_gap"] = df_rank["分数"] - df_rank["最低分"]
df_rank["rank_abs"] = (df_rank["percentile"] * total_n).astype(int)
df_rank["rank_gap"] = df_rank["rank_abs"] - df_rank["最低位次"]

print(f"衍生特征构造完成，数据形状: {df_rank.shape}")
print(f"可用列: {list(df_rank.columns)}")

# 5. 选择特征和目标 - 使用实际存在的列
FEATURE_COLS = [
    "分数", "人数", "累计人数",  # 使用"人数"替代"本段人数"
    "批次_code", "最低分", "最低位次", "平均分",
    "score_gap", "rank_gap"
]
TARGET_COL = "percentile"

# 检查所有列是否存在
missing_cols = [col for col in FEATURE_COLS + [TARGET_COL] if col not in df_rank.columns]
if missing_cols:
    print(f"❌ 缺失列: {missing_cols}")
    print("使用可用的数值列作为特征...")
    # 使用所有可用的数值列
    numeric_cols = df_rank.select_dtypes(include=['number']).columns.tolist()
    numeric_cols.remove(TARGET_COL)  # 移除目标列
    FEATURE_COLS = numeric_cols
    print(f"使用特征列: {FEATURE_COLS}")

# 数据准备
X = df_rank[FEATURE_COLS].astype("float32").values
y = df_rank[TARGET_COL].astype("float32").values

print(f"✅ 数据准备完成")
print(f"特征数据 X 形状: {X.shape}")
print(f"目标数据 y 形状: {y.shape}")
print(f"特征列: {FEATURE_COLS}")

# 6. 划分、标准化
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"训练集: {X_train_s.shape[0]} 样本")
print(f"测试集: {X_test_s.shape[0]} 样本")

# 转换为 PyTorch 张量并重塑为 LSTM 格式
X_train_tensor = torch.FloatTensor(X_train_s).unsqueeze(1)  # (samples, 1, features)
X_test_tensor = torch.FloatTensor(X_test_s).unsqueeze(1)
y_train_tensor = torch.FloatTensor(y_train)
y_test_tensor = torch.FloatTensor(y_test)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, pin_memory=USE_CUDA
)
test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, pin_memory=USE_CUDA
)

# 7. 构建 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # 取最后一个时间步
        x = self.relu(self.fc1(last_output))
        x = self.fc2(x)
        return x

n_feat = X_train_tensor.shape[-1]
model = LSTMModel(input_size=n_feat, hidden_size=64, output_size=1).to(device)

print("模型结构:")
print(model)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数（带早停）
def train_with_early_stopping(model, train_loader, test_loader, epochs=200, patience=10):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=USE_CUDA)
            batch_y = batch_y.to(device, non_blocking=USE_CUDA)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device, non_blocking=USE_CUDA)
                batch_y = batch_y.to(device, non_blocking=USE_CUDA)
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(test_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"早停在第 {epoch+1} 轮")
            break

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    return train_losses, val_losses

# 训练模型
print("开始训练 LSTM 模型...")
train_losses, val_losses = train_with_early_stopping(model, train_loader, test_loader)

# 8. 评估
model.eval()
with torch.no_grad():
    y_pred = model(
        X_test_tensor.to(device, non_blocking=USE_CUDA)
    ).squeeze().cpu().numpy()

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 模型评估结果:")
print(f"Test MAE: {mae:.4f}")
print(f"R²      : {r2:.4f}")

# 9. 保存模型与 scaler
MODEL_DIR = DATA_DIR / "lstm_rank_opt"
MODEL_DIR.mkdir(exist_ok=True)

torch.save(model.state_dict(), MODEL_DIR / "lstm_rank_opt.pth")
joblib.dump(scaler, MODEL_DIR / "scaler_rank_opt.pkl")

print(f"✅ 优化后的模型与 scaler 已保存至：{MODEL_DIR}")
