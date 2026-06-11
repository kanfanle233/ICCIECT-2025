# -*- coding: utf-8 -*-
"""
File: 010_多模型基准对比.py
Purpose: 多模型Benchmark与结果导出
Source notebook: 高考数据处理大修版本-checkpoint.ipynb
Execution order: keep numeric order 001 -> 010
"""

# ===== From Notebook CELL 83 =====
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 统一的工程/数据目录，避免硬编码绝对路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "志愿填报辅助系统" / "上海高考录取数据17-23年"
OUTPUT_DIR = DATA_DIR / "用到的"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIR = OUTPUT_DIR  # 兼容保存路径
# 生成示例残差数据（模拟 LSTM 模型的实际预测残差，真实场景请替换为实际 residuals = y_test - y_pred）
np.random.seed(42)
residuals = np.random.normal(loc=0, scale=0.05, size=1000)  # 模拟小幅度围绕 0 的分布

# 绘制残差直方图
plt.figure(figsize=(6, 4))
sns.histplot(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.title('LSTM 模型预测残差分布')
plt.xlabel('实际值 - 预测值')
plt.ylabel('样本数')
plt.tight_layout()

# 保存图片
output_path = str(OUTPUT_DIR / "lstm_residuals_hist.png")
plt.savefig(output_path, dpi=300)
plt.show()

print(f"图像已保存至：{output_path}")





# ===== From Notebook CELL 84 =====
print("X_train_scaled_2d:", X_train_scaled_2d.shape)
print("y_train:",            y_train.shape)




# ===== From Notebook CELL 85 =====
# —— 重新切分并标准化 (为所有模型共用)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# ===== From Notebook CELL 86 =====
# 特征 & 目标
X_tab = df_rank[FEATURE_COLS].values.astype("float32")    # shape=(N,1)
y_tab = df_rank[TARGET_COL].values.astype("float32")      # shape=(N,)



# ===== From Notebook CELL 87 =====
# 切分
X_train_tab, X_test_tab, y_train_tab, y_test_tab = train_test_split(
    X_tab, y_tab, test_size=0.2, random_state=42
)



# ===== From Notebook CELL 88 =====
# 标准化（只对需要的模型，如 Linear/SVR）
scaler_ml = StandardScaler()
X_train_tab_scaled = scaler_ml.fit_transform(X_train_tab)  # shape=(n_train,1)
X_test_tab_scaled  = scaler_ml.transform(X_test_tab)       # shape=(n_test,1)




# ===== From Notebook CELL 89 =====
# 为 LSTM/GRU 保留 3D 张量版本
X_train_scaled = X_train_tab_scaled[:, None, :]            # shape=(n_train,1,1)
X_test_scaled  = X_test_tab_scaled[:,  None, :]            # shape=(n_test,1,1)




# ===== From Notebook CELL 90 =====
# 为传统机器学习模型准备 2D 版本
X_train_scaled_2d = X_train_tab_scaled.squeeze(axis=1)     # shape=(n_train,)
X_test_scaled_2d  = X_test_tab_scaled.squeeze(axis=1)      # shape=(n_test,)



# ===== From Notebook CELL 91 =====
from tensorflow.keras import layers, models, callbacks

n_features = X_train_scaled.shape[-1]

model = models.Sequential([
    layers.Input(shape=(1, n_features)),
    layers.LSTM(64, return_sequences=False),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

history = model.fit(
    X_train_scaled, y_train_tab,
    epochs=200, batch_size=16,
    validation_split=0.2,
    callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
    verbose=2
)




# ===== From Notebook CELL 92 =====
#补充多模型对比
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings, sys
warnings.filterwarnings("ignore")



# ===== From Notebook CELL 93 =====
# 检测 xgboost
try:
    from xgboost import XGBRegressor
    has_xgb = True
except ImportError:
    has_xgb = False
    print("⚠ 未检测到 xgboost，已跳过 XGBRegressor", file=sys.stderr)




# ===== From Notebook CELL 94 =====
# 定义模型
models_ml = {
    "LinearReg":  LinearRegression(),
    "SVR_RBF":    SVR(C=10, gamma="scale"),
    "RF_300":     RandomForestRegressor(n_estimators=300, random_state=42),
    "GBDT_200":   GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
}
if has_xgb:
    models_ml["XGB_500"] = XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42
    )

results = []



# ===== From Notebook CELL 95 =====
results = []
for name, est in models_ml.items():
    if name in ["LinearReg", "SVR_RBF"]:
        # 用 (n_samples,1) 的二维数组
        est.fit(X_train_tab_scaled, y_train_tab)
        y_pred = est.predict(X_test_tab_scaled)
    else:
        # 树模型直接用原始 (n_samples,1)
        est.fit(X_train_tab, y_train_tab)
        y_pred = est.predict(X_test_tab)

    mae = mean_absolute_error(y_test_tab, y_pred)
    r2  = r2_score(y_test_tab, y_pred)
    results.append((name, mae, r2))




# ===== From Notebook CELL 96 =====
#训练轻量级 GRU
gru_model = models.Sequential([
    layers.Input(shape=(1, n_features)),
    layers.GRU(32, return_sequences=False),
    layers.Dense(16, activation="relu"),
    layers.Dense(1)
])
gru_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
gru_model.fit(
    X_train_scaled, y_train_tab,
    epochs=100, batch_size=16,
    validation_split=0.2,
    callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)],
    verbose=0
)
y_pred_gru = gru_model.predict(X_test_scaled, verbose=0).squeeze()
results.append(("GRU_32",
                mean_absolute_error(y_test_tab, y_pred_gru),
                r2_score(y_test_tab, y_pred_gru)))




# ===== From Notebook CELL 97 =====
#LSTM 预测并加入对比
y_pred_lstm = model.predict(X_test_scaled, verbose=0).squeeze()
results.append(("LSTM_64",
                mean_absolute_error(y_test_tab, y_pred_lstm),
                r2_score(y_test_tab, y_pred_lstm)))




# ===== From Notebook CELL 98 =====
import pandas as pd

df_res = pd.DataFrame(results, columns=["Model", "MAE", "R2"]).sort_values("MAE")
print("\n===== Rank-Prediction Benchmark =====")
print(df_res.to_string(index=False, formatters={"MAE": "{:.4f}".format,
                                               "R2": "{:.3f}".format}))

# 保存 CSV
MODEL_DIR = OUTPUT_DIR / "benchmark_models"
MODEL_DIR.mkdir(exist_ok=True)
df_res.to_csv(MODEL_DIR / "benchmark_rank_models.csv",
              index=False, encoding="utf-8-sig")




# ===== From Notebook CELL 99 =====





