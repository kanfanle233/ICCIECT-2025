# -*- coding: utf-8 -*-
"""
File: 005_优化版LSTM训练_自动GPU_CPU.py
Purpose: 优化版LSTM流程（含CUDA自动切换）
Source notebook: 高考数据处理大修版本-checkpoint.ipynb
Execution order: keep numeric order 001 -> 010
"""

# ===== From Notebook CELL 50 =====
import pandas as pd
from pathlib import Path
# 统一的工程/数据目录，避免硬编码绝对路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "志愿填报辅助系统" / "上海高考录取数据17-23年"
OUTPUT_DIR = DATA_DIR / "用到的"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIR = OUTPUT_DIR  # 兼容后续保存路径
# 1. 路径定义 - 修正为正确的路径
BASE_DIR = PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "志愿填报辅助系统" / "上海高考录取数据17-23年"
PATH_MAJOR = DATA_DIR / "2023上海专业分数线_clean.csv"

# 2. 读取
df_major = pd.read_csv(PATH_MAJOR, encoding="utf-8-sig")

print(f"原始数据形状: {df_major.shape}")
print(f"原始列名: {list(df_major.columns)}")

# 原始列：年份、生源地、科类、批次、院校名称、专业名称、
#         专业备注、最低分、最低位次、最高分、平均分

# 3. 去掉无关字段，只保留：
#    [年份、批次、最低分、最低位次、平均分]
df_major_clean = df_major[[
    "年份",
    "批次",
    "最低分",
    "最低位次",
    "平均分"
]].copy()

print(f"选择列后形状: {df_major_clean.shape}")

# 4. 把"批次" one-hot 或编码成数值（以下示例编码）
print(f"\n批次唯一值: {df_major_clean['批次'].unique()}")
df_major_clean["批次_code"] = df_major_clean["批次"].astype("category").cat.codes
print(f"批次编码映射: {dict(enumerate(df_major_clean['批次'].astype('category').cat.categories))}")

# 5. 确保数值列类型正确
for col in ["最低分", "最低位次", "平均分"]:
    df_major_clean[col] = pd.to_numeric(df_major_clean[col], errors="coerce")
    missing_count = df_major_clean[col].isna().sum()
    if missing_count > 0:
        print(f"警告: 列 '{col}' 有 {missing_count} 个缺失值")

# 删除包含缺失值的行
original_rows = len(df_major_clean)
df_major_clean = df_major_clean.dropna()
removed_rows = original_rows - len(df_major_clean)
if removed_rows > 0:
    print(f"删除了 {removed_rows} 行包含缺失值的数据")

# 6. 最终列重排序
df_major_clean = df_major_clean[[
    "年份", "批次_code", "最低分", "最低位次", "平均分"
]]

# 7. 保存或输出预览
OUT = DATA_DIR / "2023上海专业分数线_model_ready.csv"
df_major_clean.to_csv(OUT, index=False, encoding="utf-8-sig")

print(f"\n✅ 数据处理完成")
print(f"⇢ df_major_clean: {df_major_clean.shape} 行")
print(f"⇢ 保存到: {OUT}")
print("\n前5行数据:")
print(df_major_clean.head())

print(f"\n📊 数据统计:")
print(df_major_clean.describe())



# ===== From Notebook CELL 51 =====
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
BASE_DIR = PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "志愿填报辅助系统" / "上海高考录取数据17-23年"
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


# ===== From Notebook CELL 52 =====
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks



# ===== From Notebook CELL 53 =====
# 0. 路径
DIR = OUTPUT_DIR
PATH_RANK   = DIR / "上海一分一段_2017-2022_clean.csv"
PATH_MAJOR  = DIR / "2023上海专业分数线_model_ready.csv"




# ===== From Notebook CELL 54 =====
# 1. 读取数据
df_rank  = pd.read_csv(PATH_RANK,  encoding="utf-8-sig")
df_major = pd.read_csv(PATH_MAJOR, encoding="utf-8-sig")




# ===== From Notebook CELL 55 =====
# 2. 确保 df_major 只有 2023 年的那一行（如果 clean 表里还有其它年份可以先筛）
df_major_2023 = df_major.copy()  # 已经是 2023 年数据



# ===== From Notebook CELL 56 =====
# 3. 把专业表的数值特征“广播”到所有分数行
#    直接将那一行的特征附到 df_rank
#    （无需 on='年份'，因为始终用 2023 年数据）
for col in ["批次_code", "最低分", "最低位次", "平均分"]:
    df_rank[col] = df_major_2023[col].iloc[0]



# ===== From Notebook CELL 57 =====
# 4. 构造衍生特征
total_n = df_rank["累计人数"].max()
df_rank["score_gap"] = df_rank["分数"] - df_rank["最低分"]
df_rank["rank_abs"]  = (df_rank["percentile"] * total_n).astype(int)
df_rank["rank_gap"]  = df_rank["rank_abs"] - df_rank["最低位次"]



# ===== From Notebook CELL 58 =====
# 5. 选择特征和目标
FEATURE_COLS = [
    "分数", "本段人数", "累计人数",
    "批次_code", "最低分", "最低位次", "平均分",
    "score_gap", "rank_gap"
]
TARGET_COL = "percentile"

X = df_rank[FEATURE_COLS].astype("float32").values
y = df_rank[TARGET_COL].astype("float32").values




# ===== From Notebook CELL 59 =====
# 6. 划分、标准化、reshape
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)[:, None, :]
X_test_s  = scaler.transform(X_test)[:,  None, :]




# ===== From Notebook CELL 60 =====
# 7. 构建 & 训练 LSTM
n_feat = X_train_s.shape[-1]
model = models.Sequential([
    layers.Input(shape=(1, n_feat)),
    layers.LSTM(64),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])
model.compile("adam", loss="mse", metrics=["mae"])
early = callbacks.EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(
    X_train_s, y_train,
    epochs=200, batch_size=16,
    validation_split=0.2,
    callbacks=[early], verbose=2
)




# ===== From Notebook CELL 61 =====
from sklearn.metrics import mean_absolute_error, r2_score

# 8. 评估
y_pred = model.predict(X_test_s)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"Test MAE: {mae:.4f}")
print(f"R²      : {r2:.4f}")



# ===== From Notebook CELL 62 =====
# 9. 保存模型与 scaler
MODEL_DIR = DIR / "lstm_rank_opt"
MODEL_DIR.mkdir(exist_ok=True)
model.save(MODEL_DIR / "lstm_rank_opt.h5")
pd.to_pickle(scaler, MODEL_DIR / "scaler_rank_opt.pkl")
print("优化后的模型与 scaler 已保存至：", MODEL_DIR)



# ===== From Notebook CELL 63 =====
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
import numpy as np

# a) 把 LSTM 外包成一个 sklearn 可调用模型（或用简单模型做示例）
#    这里用线性回归示例，LSTM 可用 KerasClassifierWrapper / 自定义 CV
from sklearn.linear_model import LinearRegression

X_flat = X.reshape(X.shape[0], -1)  # LSTM 的 (n,1,f) → (n,f)
y      = y

# b) 5 折交叉验证：MAE 和 R²
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []
r2_scores  = []

for train_idx, test_idx in kf.split(X_flat):
    X_tr, X_te = X_flat[train_idx], X_flat[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    
    # 标准化
    from sklearn.preprocessing import StandardScaler
    scaler_cv = StandardScaler().fit(X_tr)
    X_tr_s = scaler_cv.transform(X_tr)
    X_te_s = scaler_cv.transform(X_te)
    
    # 训练简单模型（或你的 LSTM：在这里请用适配 CV 的方式）
    model_cv = LinearRegression().fit(X_tr_s, y_tr)
    
    # 评估
    y_pred = model_cv.predict(X_te_s)
    mae_scores.append(mean_absolute_error(y_te, y_pred))
    r2_scores.append(r2_score(y_te, y_pred))

print("5-Fold CV MAE:", np.mean(mae_scores), "±", np.std(mae_scores))
print("5-Fold CV R²:",  np.mean(r2_scores),  "±", np.std(r2_scores))





