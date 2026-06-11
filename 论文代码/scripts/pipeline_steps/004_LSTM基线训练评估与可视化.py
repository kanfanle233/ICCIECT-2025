# -*- coding: utf-8 -*-
"""
File: 004_LSTM基线训练评估与可视化.py
Purpose: LSTM基线训练、评估和结果图
Source notebook: 高考数据处理大修版本-checkpoint.ipynb
Execution order: keep numeric order 001 -> 010
"""

from pathlib import Path

# 统一的工程/数据目录，避免硬编码绝对路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from gaokao_recommender.paths import *

DATA_DIR = DATA_RAW_DIR
OUTPUT_DIR = DATA_PROCESSED_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIR = OUTPUT_DIR  # 兼容后续保存路径
# ===== From Notebook CELL 37 =====
# 2. 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# ===== From Notebook CELL 38 =====
# 3. 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)



# ===== From Notebook CELL 39 =====
# 4. 重塑为 LSTM 要求的 3D 张量 (samples, timesteps, features)
#    这里直接用 timesteps=1
X_train_scaled = X_train_scaled[:, None, :]
X_test_scaled  = X_test_scaled[:,  None, :]



# ===== From Notebook CELL 40 =====
# 5. 构建 LSTM 模型
n_features = X_train_scaled.shape[-1]
model = models.Sequential([
    layers.Input(shape=(1, n_features)),
    layers.LSTM(64, return_sequences=False),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)  # 回归输出：percentile
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()




# ===== From Notebook CELL 41 =====
# 6. 训练（带早停）
early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)
history = model.fit(
    X_train_scaled, y_train,
    epochs=200,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=2
)



# ===== From Notebook CELL 42 =====
# 8. 保存模型和 scaler
MODEL_DIR = DIR / "lstm_rank_model"
MODEL_DIR.mkdir(exist_ok=True)
model.save(MODEL_DIR / "lstm_rank.h5")
pd.to_pickle(scaler, MODEL_DIR / "scaler_rank.pkl")
print(f"✔ 模型和 scaler 已保存到：{MODEL_DIR}")



# ===== From Notebook CELL 43 =====
#可视化
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rcParams

# -------- 中文字体 --------
# 跨平台中文字体配置
import platform
if platform.system() == 'Windows':
    font_path = "C:/Windows/Fonts/simhei.ttf"
elif platform.system() == 'Darwin':  # macOS
    font_path = "/System/Library/Fonts/STHeiti Medium.ttc"
else:  # Linux
    font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"

try:
    simhei_font = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = simhei_font.get_name()
except:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示



# ===== From Notebook CELL 44 =====
# —— 1. 训练 & 验证 损失 曲线 —— 
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('训练与验证损失曲线')
plt.legend()
plt.tight_layout()
plt.savefig(DIR / "lstm_loss_curve.png", dpi=300)
plt.close()




# ===== From Notebook CELL 45 =====
# —— 2. 训练 & 验证 MAE 曲线 —— 
plt.figure(figsize=(8, 4))
plt.plot(history.history['mae'], label='训练 MAE')
plt.plot(history.history['val_mae'], label='验证 MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('训练与验证 MAE 曲线')
plt.legend()
plt.tight_layout()
plt.savefig(DIR / "lstm_mae_curve.png", dpi=300)
plt.close()



# ===== From Notebook CELL 46 =====
# —— 3. 实际 vs 预测 散点图 —— 
# 预测
y_pred = model.predict(X_test_scaled)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, s=10)
# 对角参考线
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val])
plt.xlabel('实际 percentile')
plt.ylabel('预测 percentile')
plt.title('实际 vs 预测 位次百分比')
plt.tight_layout()
plt.savefig(DIR / "lstm_scatter_actual_vs_pred.png", dpi=300)
plt.close()



# ===== From Notebook CELL 47 =====
from sklearn.metrics import r2_score

# —— 假设前面已经计算好了 y_pred, 并且 y_test 可用 —— 

# 4. 计算并打印 R-squared（R方值）
r_squared = r2_score(y_test, y_pred)
print(f"R-squared (R方) = {r_squared:.2f}")

print("\n✓ 数据可视化内容已添加完成。")




# ===== From Notebook CELL 48 =====
from sklearn.metrics import mean_absolute_percentage_error

# 残差直方图
residuals = y_test - y_pred.ravel()
plt.figure(figsize=(6,4))
plt.hist(residuals, bins=30, edgecolor='k')
plt.title('预测残差分布')
plt.xlabel('实际 - 预测')
plt.ylabel('样本数')
plt.tight_layout()
plt.close()

# MAPE
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE = {mape:.2%}")





