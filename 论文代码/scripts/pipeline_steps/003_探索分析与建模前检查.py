# -*- coding: utf-8 -*-
"""
File: 003_探索分析与建模前检查.py
Purpose: EDA、可视化、特征检查与调试
Source notebook: 高考数据处理大修版本-checkpoint.ipynb
Execution order: keep numeric order 001 -> 010
"""

# ===== From Notebook CELL 20 =====
import pandas as pd
from pathlib import Path

# 统一的工程/数据目录，避免硬编码绝对路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from gaokao_recommender.paths import *

DATA_DIR = DATA_RAW_DIR
OUTPUT_DIR = DATA_PROCESSED_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIR = OUTPUT_DIR  # 兼容后续保存语句

# 统一加载 002 预处理后的结果，避免重复清洗造成口径不一致
PATH_COMBO_CLEAN = OUTPUT_DIR / "subject_combo_to_mbti_clean.csv"
PATH_MAJOR_CLEAN = OUTPUT_DIR / "2023上海专业分数线_clean.csv"
PATH_SCORE_CLEAN = OUTPUT_DIR / "2023年考生高考成绩分布表_clean.csv"

for required_file in [PATH_COMBO_CLEAN, PATH_MAJOR_CLEAN, PATH_SCORE_CLEAN]:
    if not required_file.exists():
        raise FileNotFoundError(f"缺少预处理结果: {required_file}，请先运行 002_清洗标准化与核心表生成.py")

df_combo = pd.read_csv(PATH_COMBO_CLEAN, encoding="utf-8-sig")
df_major = pd.read_csv(PATH_MAJOR_CLEAN, encoding="utf-8-sig")
df_score = pd.read_csv(PATH_SCORE_CLEAN, encoding="utf-8-sig")

print("已加载清洗数据:")
print(f"df_combo: {df_combo.shape}")
print(f"df_major: {df_major.shape}")
print(f"df_score: {df_score.shape}")
print(f"分数范围: {df_score['分数'].min()} - {df_score['分数'].max()}")


# ===== From Notebook CELL 24 =====
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





# ===== From Notebook CELL 25 =====
import pandas as pd
from pathlib import Path
# 统一的工程/数据目录，避免硬编码绝对路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from gaokao_recommender.paths import *

DATA_DIR = DATA_RAW_DIR
OUTPUT_DIR = DATA_PROCESSED_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIR = OUTPUT_DIR  # 兼容后续保存语句
# 修正后的路径定义
BASE_DIR = PROJECT_ROOT
DATA_DIR = DATA_RAW_DIR

# 读取清理后的数据文件
df_score = pd.read_csv(OUTPUT_DIR / "2023年考生高考成绩分布表_clean.csv", encoding="utf-8-sig")
df_major = pd.read_csv(OUTPUT_DIR / "2023上海专业分数线_clean.csv", encoding="utf-8-sig")
df_combo = pd.read_csv(OUTPUT_DIR / "subject_combo_to_mbti_clean.csv", encoding="utf-8-sig")

# 显示数据形状验证
print("读取清理数据完成！")
print(f"df_score 形状: {df_score.shape}")
print(f"df_major 形状: {df_major.shape}")
print(f"df_combo 形状: {df_combo.shape}")



# ===== From Notebook CELL 26 =====
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
# 统一的工程/数据目录，避免硬编码绝对路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from gaokao_recommender.paths import *

DATA_DIR = DATA_RAW_DIR
OUTPUT_DIR = DATA_PROCESSED_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIR = OUTPUT_DIR  # 兼容后续保存语句
# 确保已经定义了 DATA_DIR
BASE_DIR = PROJECT_ROOT
DATA_DIR = DATA_RAW_DIR

# 分数段分布（折线 + 累计人数）
plt.figure(figsize=(10,6))
sns.lineplot(x='分数', y='人数', data=df_score, label='各分数段人数')
sns.lineplot(x='分数', y='累计人数', data=df_score, label='累计人数')
plt.title('2023 上海考生分数段分布')
plt.xlabel('高考分数')
plt.ylabel('人数')
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "score_line_2023.png", dpi=300)
plt.close()



# ===== From Notebook CELL 27 =====
# 改进版本（可选）
plt.figure(figsize=(12,6))
sns.lineplot(x='分数', y='人数', data=df_score, label='各分数段人数', linewidth=2)
sns.lineplot(x='分数', y='累计人数', data=df_score, label='累计人数', linewidth=2)
plt.title('2023 上海考生分数段分布', fontsize=14, fontweight='bold')
plt.xlabel('高考分数', fontsize=12)
plt.ylabel('人数', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "score_line_2023.png", dpi=300, bbox_inches='tight')
plt.close()



# ===== From Notebook CELL 28 =====
#专业最低分箱线图（可看整体难度水平）
plt.figure(figsize=(12,6))
sns.boxplot(x='批次', y='最低分', data=df_major, palette='Set3')
plt.title('不同批次专业最低分分布')
plt.xlabel('批次')
plt.ylabel('最低分')
plt.tight_layout()
plt.savefig(FIGURE_DIR / "major_box_2023.png", dpi=300)  # ← 保存图片
plt.close()




# ===== From Notebook CELL 29 =====
# 选科组合 × MBTI 人数柱状图（安全版本）
plt.figure(figsize=(12,6))

# 检查列名并映射
print("df_combo 的列名:", list(df_combo.columns))

# 根据实际列名选择正确的列
if 'count' in df_combo.columns:
    count_col = 'count'
elif '人数' in df_combo.columns:
    count_col = '人数'
else:
    # 如果没有人数列，使用第一个数值列
    numeric_cols = df_combo.select_dtypes(include=['number']).columns
    count_col = numeric_cols[0] if len(numeric_cols) > 0 else df_combo.columns[2]

if 'subject_combo' in df_combo.columns:
    combo_col = 'subject_combo'
elif '选科组合' in df_combo.columns:
    combo_col = '选科组合'
else:
    combo_col = df_combo.columns[0]

if 'mbti' in df_combo.columns:
    mbti_col = 'mbti'
elif 'MBTI' in df_combo.columns:
    mbti_col = 'MBTI'
else:
    mbti_col = df_combo.columns[1]

print(f"使用列: 组合={combo_col}, MBTI={mbti_col}, 人数={count_col}")

top10 = df_combo.nlargest(10, count_col)
sns.barplot(x=combo_col, y=count_col, hue=mbti_col, data=top10, dodge=False)
plt.title('人数最多的 10 个选科组合及对应 MBTI')
plt.xlabel('选科组合')
plt.ylabel('人数')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(FIGURE_DIR / "combo_mbti_bar_2023.png", dpi=300)
plt.close()



# ===== From Notebook CELL 31 =====
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks




# ===== From Notebook CELL 32 =====
from pathlib import Path
import pandas as pd

# 修正路径定义
BASE_DIR = PROJECT_ROOT
DATA_DIR = DATA_RAW_DIR

# 先检查有哪些可用的清理数据文件
print("可用的清理数据文件:")
clean_files = list(DATA_DIR.glob("*clean.csv"))
for file in clean_files:
    print(f"  📄 {file.name}")

if clean_files:
    # 使用第一个找到的清理文件
    PATH_RANK_CLEAN = clean_files[0]
    print(f"\n使用文件: {PATH_RANK_CLEAN}")

    df_rank = pd.read_csv(PATH_RANK_CLEAN, encoding="utf-8-sig")
    print(f"✅ 数据加载成功！形状: {df_rank.shape}")
else:
    print("❌ 未找到清理数据文件")
    # 使用原始成绩分布文件
    PATH_RANK = DATA_DIR / "2023年考生高考成绩分布表（上海市）.txt"
    df_rank = pd.read_csv(PATH_RANK, sep="\t", encoding="utf-8-sig")
    print(f"使用原始文件: {PATH_RANK}")
    print(f"✅ 数据加载成功！形状: {df_rank.shape}")



# ===== From Notebook CELL 33 =====
import pandas as pd
import numpy as np
from pathlib import Path
# 统一的工程/数据目录，避免硬编码绝对路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from gaokao_recommender.paths import *

DATA_DIR = DATA_RAW_DIR
OUTPUT_DIR = DATA_PROCESSED_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIR = OUTPUT_DIR  # 兼容后续保存语句
# 1. 数据加载
BASE_DIR = PROJECT_ROOT
DATA_DIR = DATA_RAW_DIR

# 加载专业分数线数据
df_major = pd.read_csv(OUTPUT_DIR / "2023上海专业分数线_clean.csv", encoding="utf-8-sig")
print(f"原始数据形状: {df_major.shape}")

# 2. 选择预测任务
print("=== 选择预测任务 ===")

# 使用专业分数线数据进行位次→分数预测
FEATURE_COLS = ["最低位次"]
TARGET_COL = "最低分"
data_source = df_major

print(f"✅ 选择：{FEATURE_COLS[0]} → {TARGET_COL} 预测")
print(f"数据源: {type(data_source)}")
print(f"选择的列: {FEATURE_COLS + [TARGET_COL]}")

# 3. 详细数据检查
print("\n=== 详细数据检查 ===")

for col in FEATURE_COLS + [TARGET_COL]:
    print(f"\n检查列 '{col}':")
    print(f"  数据类型: {data_source[col].dtype}")
    print(f"  唯一值示例: {data_source[col].unique()[:10]}")

    # 检查非数字值
    non_numeric = data_source[data_source[col].apply(lambda x: not isinstance(x, (int, float)) and not pd.isna(x))]
    if len(non_numeric) > 0:
        print(f"  非数字值: {non_numeric[col].unique()[:10]}")  # 只显示前10个

# 4. 数据清理
print(f"\n🧹 执行数据清理...")
data_clean = data_source.copy()

for col in FEATURE_COLS + [TARGET_COL]:
    # 转换为数值，非数字转为 NaN
    data_clean[col] = pd.to_numeric(data_clean[col], errors='coerce')

# 删除包含 NaN 的行
data_clean = data_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"清理前: {data_source.shape[0]} 行")
print(f"清理后: {data_clean.shape[0]} 行")

# 5. 数据准备
if len(data_clean) > 0:
    X = data_clean[FEATURE_COLS].values.astype("float32")
    y = data_clean[TARGET_COL].values.astype("float32")

    print(f"\n✅ 数据准备完成")
    print(f"特征数据 X 形状: {X.shape}")
    print(f"目标数据 y 形状: {y.shape}")
    print(f"特征范围: {X.min():.1f} - {X.max():.1f}")
    print(f"目标范围: {y.min():.1f} - {y.max():.1f}")

    # 显示数据统计
    print(f"\n📊 数据统计:")
    print(f"特征均值: {X.mean():.1f} ± {X.std():.1f}")
    print(f"目标均值: {y.mean():.1f} ± {y.std():.1f}")
else:
    print("❌ 清理后没有有效数据")
    X, y = None, None



# ===== From Notebook CELL 34 =====
import pandas as pd
import numpy as np
from pathlib import Path
# 统一的工程/数据目录，避免硬编码绝对路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from gaokao_recommender.paths import *

DATA_DIR = DATA_RAW_DIR
OUTPUT_DIR = DATA_PROCESSED_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIR = OUTPUT_DIR  # 兼容后续保存语句
# 1. 数据加载
BASE_DIR = PROJECT_ROOT
DATA_DIR = DATA_RAW_DIR

# 加载成绩分布数据
df_score = pd.read_csv(OUTPUT_DIR / "2023年考生高考成绩分布表_clean.csv", encoding="utf-8-sig")
print(f"成绩分布数据形状: {df_score.shape}")

# 2. 选择预测任务 - 分数预测百分位
FEATURE_COLS = ["分数"]
TARGET_COL = "percentile"
data_source = df_score

print(f"✅ 选择：{FEATURE_COLS[0]} → {TARGET_COL} 预测")
print(f"数据源: {type(data_source)}")

# 3. 数据准备（成绩分布数据应该是干净的）
X = data_source[FEATURE_COLS].values.astype("float32")
y = data_source[TARGET_COL].values.astype("float32")

print(f"\n✅ 数据准备完成")
print(f"特征数据 X 形状: {X.shape}")
print(f"目标数据 y 形状: {y.shape}")
print(f"分数范围: {X.min():.1f} - {X.max():.1f}")
print(f"百分位范围: {y.min():.4f} - {y.max():.4f}")



# ===== From Notebook CELL 35 =====
# 详细调试版本
print("=== 详细数据检查 ===")

# 检查数据中的问题值
print(f"数据源: {type(data_source)}")
print(f"选择的列: {FEATURE_COLS + [TARGET_COL]}")

for col in FEATURE_COLS + [TARGET_COL]:
    print(f"\n检查列 '{col}':")
    print(f"  数据类型: {data_source[col].dtype}")
    print(f"  唯一值示例: {data_source[col].unique()[:10]}")

    # 检查非数字值
    non_numeric = data_source[data_source[col].apply(lambda x: not isinstance(x, (int, float)) and not pd.isna(x))]
    if len(non_numeric) > 0:
        print(f"  非数字值: {non_numeric[col].unique()}")

# 数据清理
print(f"\n🧹 执行数据清理...")
data_clean = data_source.copy()

for col in FEATURE_COLS + [TARGET_COL]:
    # 转换为数值，非数字转为 NaN
    data_clean[col] = pd.to_numeric(data_clean[col], errors='coerce')

# 删除包含 NaN 的行
data_clean = data_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"清理前: {data_source.shape[0]} 行")
print(f"清理后: {data_clean.shape[0]} 行")

# 数据准备
if len(data_clean) > 0:
    X = data_clean[FEATURE_COLS].values.astype("float32")
    y = data_clean[TARGET_COL].values.astype("float32")

    print(f"\n✅ 数据准备完成")
    print(f"特征数据 X 形状: {X.shape}")
    print(f"目标数据 y 形状: {y.shape}")
else:
    print("❌ 清理后没有有效数据")
    X, y = None, None



# ===== From Notebook CELL 36 =====
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

print("开始 MLP 模型训练...")

# 1. 数据准备（确保 X 和 y 已经定义）
print(f"特征数据 X 形状: {X.shape}")
print(f"目标数据 y 形状: {y.shape}")

# 2. 数据分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"训练集: {X_train.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")

# 3. 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ 数据预处理完成")

# 4. 构建 MLP 模型
mlp_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),  # 两个隐藏层：64个神经元和32个神经元
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10,
    verbose=True
)

# 5. 训练模型
print("训练 MLP 模型...")
mlp_model.fit(X_train_scaled, y_train)

# 6. 模型评估
y_pred = mlp_model.predict(X_test_scaled)

print("\n📊 MLP 模型评估结果:")
print(f"MAE (平均绝对误差): {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE (均方根误差): {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# 7. 保存模型
BASE_DIR = PROJECT_ROOT
DATA_DIR = DATA_RAW_DIR
MODEL_DIR = DATA_DIR / "mlp_rank_model"
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(mlp_model, MODEL_DIR / "mlp_rank.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler_rank.pkl")

print(f"✅ MLP 模型和 scaler 已保存到：{MODEL_DIR}")

# 8. 可视化预测结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('MLP 模型预测 vs 实际值')
plt.tight_layout()
plt.savefig(MODEL_DIR / "prediction_scatter.png", dpi=300)
plt.close()

# 9. 显示训练历史（如果可用）
if hasattr(mlp_model, 'loss_curve_'):
    plt.figure(figsize=(10, 4))
    plt.plot(mlp_model.loss_curve_)
    plt.title('MLP 训练损失曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "training_loss.png", dpi=300)
    plt.close()





