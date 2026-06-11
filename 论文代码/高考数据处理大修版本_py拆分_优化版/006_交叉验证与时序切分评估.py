# -*- coding: utf-8 -*-
"""
File: 006_交叉验证与时序切分评估.py
Purpose: 补充评估：KFold/时序切分/对比图
Source notebook: 高考数据处理大修版本-checkpoint.ipynb
Execution order: keep numeric order 001 -> 010
"""

# ===== From Notebook CELL 65 =====
from pathlib import Path
# 统一的工程/数据目录，避免硬编码绝对路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "志愿填报辅助系统" / "上海高考录取数据17-23年"
OUTPUT_DIR = DATA_DIR / "用到的"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIR = OUTPUT_DIR  # 兼容后续保存路径
# 把下面这一行替换成你本地的完整目录
DIR = OUTPUT_DIR

# 验证一下目录里到底有哪些文件
print(list(DIR.iterdir()))




# ===== From Notebook CELL 66 =====
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. 设定正确的目录 - 修正路径
BASE_DIR = PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "志愿填报辅助系统" / "上海高考录取数据17-23年"

# 2. 读取清洗后的分数分布表 - 使用我们生成的文件
df_rank = pd.read_csv(DATA_DIR / "2023年考生高考成绩分布表_clean.csv", encoding="utf-8-sig")

print(f"数据形状: {df_rank.shape}")
print(f"列名: {list(df_rank.columns)}")

# 检查是否有年份列，如果没有则添加（因为我们只有2023年数据）
if "年份" not in df_rank.columns:
    df_rank["年份"] = 2023
    print("添加年份列: 2023")

# 3. 由于我们只有2023年数据，使用随机分割替代时序分割
from sklearn.model_selection import train_test_split

# 数据准备
X = df_rank[["分数"]].values
y = df_rank["percentile"].values

# 随机分割训练/测试集
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"训练集: {X_tr.shape[0]} 样本")
print(f"测试集: {X_te.shape[0]} 样本")

# 4. 标准化
scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_te_s = scaler.transform(X_te)

# 5. 训练 & 评估
model = LinearRegression().fit(X_tr_s, y_tr)
y_pred = model.predict(X_te_s)

print("Test MAE:", mean_absolute_error(y_te, y_pred))
print("Test R² :", r2_score(y_te, y_pred))

# 显示模型系数
print(f"模型系数: {model.coef_[0]:.4f}")
print(f"模型截距: {model.intercept_:.4f}")

# 6. 可视化：散点图 + 残差直方图
plt.figure(figsize=(6,6))
plt.scatter(y_te, y_pred, s=15, alpha=0.7)
mn, mx = y_te.min(), y_te.max()
plt.plot([mn, mx], [mn, mx], 'r--')
plt.xlabel('真实 percentile')
plt.ylabel('预测 percentile')
plt.title('2023 年测试集：实际 vs 预测')
plt.tight_layout()
plt.savefig(DATA_DIR / "linear_regression_scatter.png", dpi=300)
plt.show()

residuals = y_te - y_pred
plt.figure(figsize=(6,4))
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('真实 – 预测')
plt.ylabel('样本数')
plt.title('2023 年测试集：残差分布')
plt.tight_layout()
plt.savefig(DATA_DIR / "linear_regression_residuals.png", dpi=300)
plt.show()

# 7. 额外可视化：预测曲线
plt.figure(figsize=(10, 6))
# 按分数排序显示
sorted_indices = np.argsort(X_te.flatten())
X_sorted = X_te[sorted_indices]
y_te_sorted = y_te[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

plt.plot(X_sorted, y_te_sorted, 'o', alpha=0.6, label='实际值', markersize=4)
plt.plot(X_sorted, y_pred_sorted, 'r-', label='预测值', linewidth=2)
plt.xlabel('分数')
plt.ylabel('percentile')
plt.title('线性回归：分数 vs percentile')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(DATA_DIR / "linear_regression_curve.png", dpi=300)
plt.show()



# ===== From Notebook CELL 67 =====
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. 设定正确的目录
DIR = OUTPUT_DIR

# 2. 读取清洗后的分数分布表
df_rank = pd.read_csv(DIR / "上海一分一段_2017-2022_clean.csv", encoding="utf-8-sig")

# 3. 随后按年份做时序切分
train = df_rank[df_rank["年份"] <= 2021]
test  = df_rank[df_rank["年份"] == 2022]

X_tr = train[["分数"]].values
y_tr = train["percentile"].values
X_te = test [ ["分数"] ].values
y_te = test [ "percentile"].values

# 4. 标准化
scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_te_s = scaler.transform(X_te)

# 5. 训练 & 评估
model = LinearRegression().fit(X_tr_s, y_tr)
y_pred = model.predict(X_te_s)

print("Time-Split Test MAE:", mean_absolute_error(y_te, y_pred))
print("Time-Split Test R² :",   r2_score(y_te, y_pred))

# 6. 可视化：散点图 + 残差直方图
plt.figure(figsize=(6,6))
plt.scatter(y_te, y_pred, s=15, alpha=0.7)
mn, mx = y_te.min(), y_te.max()
plt.plot([mn, mx], [mn, mx], 'r--')
plt.xlabel('真实 percentile'); plt.ylabel('预测 percentile')
plt.title('2022 年测试集：实际 vs 预测')
plt.tight_layout()
plt.savefig(DIR / "time_split_scatter.png", dpi=300)
plt.show()

residuals = y_te - y_pred
plt.figure(figsize=(6,4))
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('真实 – 预测'); plt.ylabel('样本数')
plt.title('2022 年测试集：残差分布')
plt.tight_layout()
plt.savefig(DIR / "time_split_residuals.png", dpi=300)
plt.show()




# ===== From Notebook CELL 68 =====
from pathlib import Path
import pandas as pd

# 1. 把 DIR 改成你本地的完整目录
DIR = OUTPUT_DIR

# 2. 读取清洗后的分数分布表
df_rank = pd.read_csv(DIR / "上海一分一段_2017-2022_clean.csv", encoding="utf-8-sig")

# 3. 读取 2023 专业分数线特征
df_major = pd.read_csv(DIR / "2023上海专业分数线_model_ready.csv", encoding="utf-8-sig")

# 确认都加载成功
print(df_rank.shape, df_major.shape)




# ===== From Notebook CELL 69 =====
# 时序切分
train = df_rank[df_rank["年份"] <= 2021]
test  = df_rank[df_rank["年份"] == 2022]

# 特征与目标
FEATURE_COLS = ["分数"]
X_train = train[FEATURE_COLS].values.reshape(-1,1)
y_train = train["percentile"].values
X_test  = test [FEATURE_COLS].values.reshape(-1,1)
y_test  = test ["percentile"].values

# 标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# 训练最简单模型做演示
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

# 可视化：散点图
import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, s=15, alpha=0.7)
minv, maxv = y_test.min(), y_test.max()
plt.plot([minv,maxv], [minv,maxv], 'r--')
plt.xlabel('真实 percentile')
plt.ylabel('预测 percentile')
plt.title('2022 测试集 实际 vs 预测')
plt.tight_layout()
plt.savefig(DIR / "time_split_scatter.png", dpi=300)
plt.show()




# ===== From Notebook CELL 71 =====
import pandas as pd
from pathlib import Path
import re

# 0. 路径
BASE_DIR = PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "志愿填报辅助系统" / "上海高考录取数据17-23年"
IN_FILE = DATA_DIR / "2023上海专业分数线_clean.csv"
OUT_FILE = DATA_DIR / "2023上海专业分数线_with_PredictedMBTI.csv"

# 1. 读取数据
df = pd.read_csv(IN_FILE, encoding="utf-8-sig")
print(f"原始数据形状: {df.shape}")

# 2. 扩展的关键词映射（基于未映射的专业进行补充）
keyword_mbti = [
    # 工程技术类
    (r"机械|制造|设计|车辆工程|材料成型|工业设计|机电|数控", "ISTJ"),
    (r"计算机|软件|信息|人工智能|数据科学|网络安全|物联网|大数据", "INTJ"),
    (r"自动化|电气|电子|通信|集成电路|能源动力|电力|控制|仪器科学", "INTP"),
    (r"建筑|土建|市政|工程管理|园林|测绘|资源勘查|水利|轨道|公路|铁路|土木工程", "ESTP"),

    # 商业管理类
    (r"电子商务|财务会计|物流|经济贸易|金融|工商管理|市场|会计|财务|审计|保险|投资", "ESTJ"),
    (r"旅游|酒店|餐饮|管理|会展经济", "ESFP"),

    # 医疗健康类
    (r"护理|临床医学|公共卫生|预防医学|医学影像|麻醉学", "ESFJ"),
    (r"医学技术|康复|护理学|口腔医学|健康管理|检验医学|影像技术", "ISFJ"),
    (r"药学|中医药|制药工程", "INFJ"),

    # 人文社科类
    (r"教育|师范|学前教育|特殊教育", "ENFJ"),
    (r"外国语言|英语|日语|法语|德语|翻译", "ENFJ"),
    (r"新闻传播|广告学|编辑出版|网络与新媒体", "ENFP"),
    (r"法学|法律|法务|司法|知识产权", "ENTJ"),
    (r"公共管理|行政管理|人力资源|社会保障", "ENTJ"),
    (r"心理学|社会学|人类学|历史学|哲学", "INFJ"),

    # 自然科学类
    (r"数学|物理|化学|统计学|天文学|地质学", "INTP"),
    (r"生物|生物技术|生命科学|生态学|生物工程", "INFJ"),

    # 艺术设计类
    (r"艺术设计|美术|音乐|舞蹈|戏剧|影视|动画|数字媒体", "ESFP"),
    (r"服装设计|产品设计|环境设计|视觉传达", "ISFP"),

    # 农业环境类
    (r"农业|林业|畜牧业|水产|园艺|植物保护", "ISFJ"),
    (r"环境工程|安全工程|食品科学|纺织工程|材料科学", "ISFJ"),

    # 新增映射（基于未映射的专业）
    (r"房地产|物业管理", "ESTJ"),
    (r"文化服务|文化产业管理", "ENFP"),
    (r"公共事业|社区管理|社会工作", "ENFJ"),
    (r"语言类|汉语国际教育|应用语言学", "INFJ"),
    (r"航空运输|飞行技术|空中乘务", "ESTP"),
    (r"铁道运输|轨道交通", "ISTJ"),
    (r"化工技术|化学工程|应用化学", "INTP"),
    (r"公共服务|公共管理", "ESFJ"),
    (r"水上运输|航海技术|轮机工程", "ISTJ"),
    (r"非金属材料|高分子材料|复合材料", "INTP"),

    # 其他
    (r"体育|运动训练|体育教育", "ESFJ"),
]

# 3. 映射函数
def predict_mbti(major_name):
    for pattern, mbti in keyword_mbti:
        if re.search(pattern, major_name):
            return mbti
    return None

# 4. 应用映射
df = df.copy()  # 避免链式赋值警告
df["Predicted_MBTI"] = df["专业名称"].apply(predict_mbti)

# 5. 分析结果
total_count = len(df)
mapped_count = df["Predicted_MBTI"].notna().sum()
unmapped_count = total_count - mapped_count

print(f"\n📊 MBTI 映射结果:")
print(f"总专业数: {total_count}")
print(f"已映射: {mapped_count} ({mapped_count/total_count*100:.1f}%)")
print(f"未映射: {unmapped_count} ({unmapped_count/total_count*100:.1f}%)")

# 显示未映射的专业
unmapped_majors = df.loc[df["Predicted_MBTI"].isnull(), "专业名称"].unique()
if len(unmapped_majors) > 0:
    print(f"\n⚠️ 未映射的专业 (前15个):")
    for major in unmapped_majors[:15]:
        print(f"  - {major}")

# 6. 填充默认值（修复警告）
df.loc[df["Predicted_MBTI"].isnull(), "Predicted_MBTI"] = "ISTJ"

# 7. 显示 MBTI 分布
print(f"\n🎯 MBTI 类型分布:")
mbti_dist = df["Predicted_MBTI"].value_counts().sort_values(ascending=False)
for mbti, count in mbti_dist.items():
    percentage = count / total_count * 100
    print(f"  {mbti}: {count:>3} 个专业 ({percentage:5.1f}%)")

# 8. 保存未映射专业列表（用于后续完善）
if len(unmapped_majors) > 0:
    unmapped_df = pd.DataFrame({"未映射专业": unmapped_majors})
    unmapped_file = DATA_DIR / "未映射MBTI的专业列表.csv"
    unmapped_df.to_csv(unmapped_file, index=False, encoding="utf-8-sig")
    print(f"\n📝 未映射专业列表已保存: {unmapped_file}")

# 9. 保存结果
df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
print(f"\n✅ 已保存带 MBTI 预测的文件: {OUT_FILE}")

# 10. 显示各 MBTI 类型的代表性专业
print(f"\n🎓 各 MBTI 类型的代表性专业:")
for mbti in mbti_dist.index[:8]:  # 显示前8个主要类型
    sample_majors = df[df["Predicted_MBTI"] == mbti]["专业名称"].unique()[:3]
    print(f"\n{mbti}:")
    for major in sample_majors:
        print(f"  - {major}")

# 11. 显示示例数据
print(f"\n📋 随机示例数据:")
sample = df[["院校名称", "专业名称", "Predicted_MBTI"]].sample(10)
for _, row in sample.iterrows():
    print(f"  {row['院校名称'][:10]}... - {row['专业名称'][:15]}... - {row['Predicted_MBTI']}")




