"""
高考志愿填报辅助系统 - 完整数据处理与模型训练流水线

本文件是从 Jupyter Notebook 导出的完整代码，包含高考数据处理的全流程：
数据读取、清洗、可视化、LSTM模型训练、SVD推荐系统、SHAP可解释性分析等。
教学重点：端到端机器学习项目的完整工作流。
"""
# %% [CELL 1]
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# %% [CELL 2]
# 正确定义文件路径
PATH_COMBO_MBTI = '志愿填报辅助系统/上海高考录取数据17-23年/subject_combo_to_mbti.txt'
PATH_MAJOR = '志愿填报辅助系统/上海高考录取数据17-23年/2023上海专业分数线.txt'
PATH_SCORE_DIST = '志愿填报辅助系统/上海高考录取数据17-23年/2023年考生高考成绩分布表（上海市）.txt'

# 读取数据
df_combo = pd.read_csv(PATH_COMBO_MBTI, sep='\t', encoding='utf-8-sig')
df_major = pd.read_csv(PATH_MAJOR, sep='\t', encoding='utf-8-sig')
df_score = pd.read_csv(PATH_SCORE_DIST, sep='\t', encoding='utf-8-sig')

# %% [CELL 3]
# 3. 对应列名（如有不同请检查 txt 头部改这里）
df_combo = df_combo.rename(columns={
    "mbti": "MBTI",
    "count": "人数count",
    "subject_combo": "选科组合"
})

# %% [CELL 4]
# 4. 按 MBTI 累加总人数，得到全体分布
mbti_pie = df_combo.groupby("MBTI")["人数count"].sum().sort_values(ascending=False)

# %% [CELL 5]
# 5. 绘制饼图
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

# 1. 指定黑体字体路径（Windows自带）
font_path = "C:/Windows/Fonts/simhei.ttf"
simhei_font = font_manager.FontProperties(fname=font_path)

# 2. 全局设置字体
plt.rcParams['font.family'] = simhei_font.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示


# 3. 直接用原先的画图代码即可
plt.figure(figsize=(7, 7))
plt.pie(
    mbti_pie,
    labels=mbti_pie.index,
    autopct="%1.1f%%",
    startangle=90,
    counterclock=False,
)
plt.title("2023年上海考生 6选3组合-MBTI类型分布")
plt.tight_layout()
OUT_FIG = Path(r"C:\Users\Lenovo\Desktop\志愿填报辅助系统\上海高考录取数据17-23年\用到的\fig_mbti_pie_2023.png")
plt.savefig(OUT_FIG, dpi=300)
plt.show()

print(f"饼图已保存：{OUT_FIG}")

# %% [CELL 6]
# 第6步中的DIR变量需要定义
# 建议添加：
import pathlib
DIR = pathlib.Path("数据路径")  # 替换为实际路径import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from pathlib import Path

# —— 1. 字体配置（Windows 黑体） ——
font_path = "C:/Windows/Fonts/simhei.ttf"
simhei_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family']        = simhei_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# —— 2. 准备数据 —— 
# 假设 mbti_pie 已经存在：
# mbti_pie = df_mbti.groupby("MBTI")["人数count"].sum()
labels = mbti_pie.index
sizes  = mbti_pie.values
total  = sizes.sum()

# —— 3. 输出目录 —— 
OUT_DIR = Path(r"C:\Users\Lenovo\Desktop\志愿填报辅助系统\上海高考录取数据17-23年\用到的")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# —— 4. Matplotlib Pastel1 柔和饼图 —— 
cmap      = plt.get_cmap('Pastel1')
colors_p1 = cmap(np.linspace(0, 1, len(labels)))

fig1, ax1 = plt.subplots(figsize=(7,7))
ax1.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90,
    counterclock=False,
    colors=colors_p1,
    wedgeprops={'edgecolor':'white'}
)
ax1.set_title("2023年上海考生 6选3组合-MBTI类型分布（Pastel1）")
plt.tight_layout()
fig1.savefig(OUT_DIR/"mbti_pie_pastel1.png", dpi=300)
plt.show()


# —— 5. Seaborn “pastel” 调色板饼图 —— 
palette = sns.color_palette("pastel", len(labels))
fig2, ax2 = plt.subplots(figsize=(7,7))
ax2.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90,
    counterclock=False,
    colors=palette,
    wedgeprops={'linewidth':0.5, 'edgecolor':'white'}
)
ax2.set_title("2023年上海考生 6选3组合-MBTI类型分布（Seaborn Pastel）")
plt.tight_layout()
fig2.savefig(OUT_DIR/"mbti_pie_seaborn_pastel.png", dpi=300)
plt.show()


# —— 6. 水平条形图 —— 
mbti_sorted = mbti_pie.sort_values(ascending=True)
colors_bar  = cmap(np.linspace(0, 1, len(mbti_sorted)))

fig3, ax3 = plt.subplots(figsize=(8,6))
ax3.barh(
    mbti_sorted.index,
    mbti_sorted.values,
    color=colors_bar
)
for value, mbti in zip(mbti_sorted.values, mbti_sorted.index):
    pct = value / total
    ax3.text(
        value + total*0.005,
        mbti,
        f"{pct:.1%}",
        va='center'
    )
ax3.set_xlabel("考生人数")
ax3.set_title("2023年上海考生 6选3组合-MBTI类型分布（水平条形）")
plt.tight_layout()
fig3.savefig(OUT_DIR/"mbti_barh_pastel1.png", dpi=300)
plt.show()


# —— 7. 环形图 (Donut) —— 
fig4, ax4 = plt.subplots(figsize=(7,7))
ax4.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90,
    counterclock=False,
    colors=colors_p1,
    wedgeprops={'width':0.4, 'edgecolor':'white'}
)
ax4.set_title("2023年上海考生 6选3组合-MBTI类型分布（环形图）")
plt.tight_layout()
fig4.savefig(OUT_DIR/"mbti_donut_pastel1.png", dpi=300)
plt.show()


print("所有图表已保存到：", OUT_DIR.resolve())


# %% [CELL 8]
# 0. 环境准备
import pandas as pd
from pathlib import Path

# %% [CELL 9]
# ★ 保证 Jupyter 的列完整显示
pd.set_option('display.max_columns', 50)

# %% [CELL 10]
DIR = Path("志愿填报辅助系统/上海高考录取数据17-23年")
PATH_COMBO = DIR / "subject_combo_to_mbti.txt"
PATH_MAJOR = DIR / "2023上海专业分数线.txt"
PATH_SCORE = DIR / "2023年考生高考成绩分布表（上海市）.txt"

# %% [CELL 11]
df_combo = pd.read_csv(PATH_COMBO, sep='\t', encoding='utf-8-sig')
df_major = pd.read_csv(PATH_MAJOR, sep='\t', encoding='utf-8-sig')
df_score = pd.read_csv(PATH_SCORE, sep='\t', encoding='utf-8-sig')

# %% [CELL 13]
# 1. 列名标准化
df_combo = df_combo.rename(columns={
    "subject_combo": "选科组合",
    "mbti": "MBTI",
    "count": "人数"
})

# %% [CELL 14]
# 2. 去重 + 缺失
df_combo = (
    df_combo
    .dropna(subset=["选科组合", "MBTI"])
    .drop_duplicates(subset=["选科组合", "MBTI"])
)

# %% [CELL 15]
# 3. 数据类型
df_combo["人数"] = (
    df_combo["人数"]
    .astype(str)
    .str.replace(r"[^\d.]", "", regex=True)  # 去掉万、w 等字母
    .replace("", "0")
    .astype(int)
)

# %% [CELL 16]
# 4. 保存
df_combo.to_csv(DIR / "subject_combo_to_mbti_clean.csv",
                index=False, encoding="utf-8-sig")


# %% [CELL 18]
# 1. 列名保持中文，删除全空列
df_major = df_major.dropna(axis=1, how="all")

# 2. 只保留 2023 年份
df_major = df_major[df_major["年份"] == 2023]

# 3. 数值字段统一转
for col in ["最低分", "最低位次", "最高分", "平均分"]:
    s = (
        df_major[col]
        .astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
    )
    df_major[col] = pd.to_numeric(s, errors="coerce")

# 4. 丢掉最低分为空的行
df_major = df_major.dropna(subset=["最低分"])

# 5. 同一院校-专业取最低最低分（最难录）
df_major = (
    df_major
    .sort_values("最低分")
    .drop_duplicates(subset=["院校名称", "专业名称"], keep="first")
    .reset_index(drop=True)
)

# 6. 保存干净表
df_major.to_csv(DIR / "2023上海专业分数线_clean.csv",
                index=False, encoding="utf-8-sig")


# %% [CELL 19]
# 处理成绩分布数据的完整代码

# 1. 首先确保列名正确
print("处理前的列名:", list(df_score.columns))

# 如果列名不是预期的，先重命名
if '分数' not in df_score.columns:
    # 根据实际情况重命名列
    df_score.columns = ['分数', '人数', '累计人数']

# 2. 去掉备注行：只保留分数列完全是数字的行
df_score = df_score[df_score["分数"].astype(str).str.match(r"^\d+$")]

# 3. 转换数据类型
df_score["分数"] = (
    df_score["分数"]
    .astype(str)
    .str.replace("分及以上", "", regex=False)
    .str.replace("分", "", regex=False)
    .astype(int)
)

for col in ["人数", "累计人数"]:
    df_score[col] = (
        df_score[col]
        .astype(str)
        .str.replace(r"[^\d]", "", regex=True)
        .astype(int)
    )

# 4. 计算百分位
total = df_score["累计人数"].iloc[-1]
df_score["percentile"] = df_score["累计人数"] / total

# 5. 保存干净表
df_score.to_csv(DIR / "2023年考生高考成绩分布表_clean.csv",
                index=False, encoding="utf-8-sig")

print("处理完成！")
print("处理后的 df_score:")
print(df_score.head())
print(f"总人数: {total}")
print(f"形状: {df_score.shape}")

# %% [CELL 20]
import pandas as pd
from pathlib import Path

# 1. 路径定义 - 修正为正确的路径
BASE_DIR = Path("E:/代码/minicondapythonProject1")
DATA_DIR = BASE_DIR / "论文代码" / "志愿填报辅助系统" / "上海高考录取数据17-23年"

# 先检查有哪些文件可用
print("目录中的文件:")
for file in DATA_DIR.iterdir():
    if file.is_file():
        print(f"  📄 {file.name}")

# 如果没有一分一段文件，我们可以使用现有的文件
# 或者搜索其他可能包含排名数据的文件
print("\n搜索可能包含排名数据的文件...")
possible_rank_files = []
for file in DATA_DIR.iterdir():
    if file.is_file() and any(keyword in file.name.lower() for keyword in ["rank", "score", "分布", "成绩"]):
        possible_rank_files.append(file)

if possible_rank_files:
    print("找到的可能包含排名数据的文件:")
    for i, file in enumerate(possible_rank_files):
        print(f"  {i+1}. {file.name}")

    # 使用第一个文件
    PATH_RANK = possible_rank_files[0]
    print(f"\n使用文件: {PATH_RANK}")
else:
    # 如果没有找到，使用成绩分布文件
    PATH_RANK = DATA_DIR / "2023年考生高考成绩分布表（上海市）.txt"
    print(f"\n使用默认文件: {PATH_RANK}")

# 2. 读取并预览列名
try:
    df_rank = pd.read_csv(PATH_RANK, sep="\t", encoding="utf-8-sig")
    print("\n原始列名：", df_rank.columns.tolist())
    print("数据形状:", df_rank.shape)
    print("\n前5行数据:")
    print(df_rank.head())

    # 3. 去除多余列（添加错误检查）
    columns_to_drop = ["省份", "科类"]
    existing_columns_to_drop = [col for col in columns_to_drop if col in df_rank.columns]
    if existing_columns_to_drop:
        df_rank = df_rank.drop(columns=existing_columns_to_drop)
        print(f"\n已删除列: {existing_columns_to_drop}")

    # 4. 清洗数值列
    # 4.1 处理分数列
    df_rank["分数"] = (
        df_rank["分数"]
        .astype(str)
        .str.replace("分及以上", "", regex=False)
        .str.extract(r"(\d+)", expand=False)
        .astype(int)
    )

    # 4.2 处理人数列
    for col in ["本段人数", "累计人数"]:
        if col in df_rank.columns:
            df_rank[col] = (
                df_rank[col]
                .astype(str)
                .str.replace(r"[^\d]", "", regex=True)
                .astype(int)
            )

    # 5. 计算百分位
    if "累计人数" in df_rank.columns:
        total = df_rank["累计人数"].iloc[-1]
        df_rank["percentile"] = df_rank["累计人数"] / total
        print(f"\n总人数: {total}")
    else:
        print("错误: '累计人数' 列不存在，无法计算百分位")

    # 6. 重排列及保存
    desired_columns = ["年份", "分数", "本段人数", "累计人数", "percentile"]
    available_columns = [col for col in desired_columns if col in df_rank.columns]
    df_rank = df_rank[available_columns]

    OUT_PATH = DATA_DIR / "上海一分一段_2017-2022_clean.csv"
    df_rank.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n✅ 已保存清洗结果：{OUT_PATH}")
    print(f"最终数据形状: {df_rank.shape}")
    print("\n前5行数据:")
    print(df_rank.head())

except Exception as e:
    print(f"处理文件时出错: {e}")
    print(f"请检查文件 {PATH_RANK} 的内容和格式")

# %% [CELL 21]
import pandas as pd
from pathlib import Path

# 1. 路径定义
BASE_DIR = Path("E:/代码/minicondapythonProject1")
DATA_DIR = BASE_DIR / "论文代码" / "志愿填报辅助系统" / "上海高考录取数据17-23年"

print("目录中的文件:")
for file in DATA_DIR.iterdir():
    if file.is_file():
        print(f"  📄 {file.name}")

# 使用成绩分布文件，这个应该是正确的
PATH_RANK = DATA_DIR / "2023年考生高考成绩分布表（上海市）.txt"
print(f"\n使用文件: {PATH_RANK}")

# 2. 读取并预览列名
try:
    df_rank = pd.read_csv(PATH_RANK, sep="\t", encoding="utf-8-sig")
    print("\n原始列名：", df_rank.columns.tolist())
    print("数据形状:", df_rank.shape)
    print("\n前10行数据:")
    print(df_rank.head(10))

    # 检查数据内容，看看是否需要重命名列
    print("\n检查数据内容...")

    # 如果列名不是预期的，先重命名
    if len(df_rank.columns) >= 3:
        df_rank.columns = ['分数', '人数', '累计人数']
        print("已重命名列为: ['分数', '人数', '累计人数']")

    # 3. 清洗数值列
    # 3.1 去掉备注行：只保留分数列完全是数字的行
    df_rank = df_rank[df_rank["分数"].astype(str).str.match(r"^\d+$")]
    print(f"过滤后数据形状: {df_rank.shape}")

    # 3.2 处理分数列
    df_rank["分数"] = (
        df_rank["分数"]
        .astype(str)
        .str.replace("分及以上", "", regex=False)
        .str.replace("分", "", regex=False)
        .astype(int)
    )

    # 3.3 处理人数列
    for col in ["人数", "累计人数"]:
        df_rank[col] = (
            df_rank[col]
            .astype(str)
            .str.replace(r"[^\d]", "", regex=True)
            .astype(int)
        )

    # 4. 计算百分位
    total = df_rank["累计人数"].iloc[-1]
    df_rank["percentile"] = df_rank["累计人数"] / total
    print(f"\n总人数: {total}")

    # 添加年份列（因为是2023年数据）
    df_rank["年份"] = 2023

    # 5. 重排列及保存
    df_rank = df_rank[["年份", "分数", "人数", "累计人数", "percentile"]]

    OUT_PATH = DATA_DIR / "上海一分一段_2023_clean.csv"
    df_rank.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n✅ 已保存清洗结果：{OUT_PATH}")
    print(f"最终数据形状: {df_rank.shape}")
    print("\n前10行数据:")
    print(df_rank.head(10))

    # 显示统计信息
    print(f"\n统计信息:")
    print(f"分数范围: {df_rank['分数'].min()} - {df_rank['分数'].max()}")
    print(f"总记录数: {len(df_rank)}")

except Exception as e:
    print(f"处理文件时出错: {e}")
    import traceback
    print(traceback.format_exc())

# %% [CELL 22]
import pandas as pd
from pathlib import Path

# 1. 路径定义
BASE_DIR = Path("E:/代码/minicondapythonProject1")
DATA_DIR = BASE_DIR / "论文代码" / "志愿填报辅助系统" / "上海高考录取数据17-23年"

# 定义文件路径
PATH_COMBO = DATA_DIR / "subject_combo_to_mbti.txt"
PATH_MAJOR = DATA_DIR / "2023上海专业分数线.txt"
PATH_SCORE = DATA_DIR / "2023年考生高考成绩分布表（上海市）.txt"

print("开始处理三个数据文件...\n")

# ==================== 处理 subject_combo_to_mbti ====================
print("1. 处理 subject_combo_to_mbti 数据...")
df_combo = pd.read_csv(PATH_COMBO, sep='\t', encoding='utf-8-sig')
print(f"  原始形状: {df_combo.shape}")
print(f"  列名: {list(df_combo.columns)}")

# 保存清理版本
df_combo.to_csv(DATA_DIR / "subject_combo_to_mbti_clean.csv", index=False, encoding='utf-8-sig')
print("  ✅ 已保存清理版本\n")

# ==================== 处理专业分数线数据 ====================
print("2. 处理专业分数线数据...")
df_major = pd.read_csv(PATH_MAJOR, sep='\t', encoding='utf-8-sig')
print(f"  原始形状: {df_major.shape}")

# 1. 列名保持中文，删除全空列
df_major = df_major.dropna(axis=1, how="all")

# 2. 只保留 2023 年份
df_major = df_major[df_major["年份"] == 2023]

# 3. 数值字段统一转换
for col in ["最低分", "最低位次", "最高分", "平均分"]:
    s = (
        df_major[col]
        .astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
    )
    df_major[col] = pd.to_numeric(s, errors="coerce")

# 4. 丢掉最低分为空的行
df_major = df_major.dropna(subset=["最低分"])

# 5. 同一院校-专业取最低最低分（最难录）
df_major = (
    df_major
    .sort_values("最低分")
    .drop_duplicates(subset=["院校名称", "专业名称"], keep="first")
    .reset_index(drop=True)
)

# 6. 保存干净表
df_major.to_csv(DATA_DIR / "2023上海专业分数线_clean.csv", index=False, encoding="utf-8-sig")
print(f"  清理后形状: {df_major.shape}")
print("  ✅ 已保存清理版本\n")

# ==================== 处理成绩分布数据 ====================
print("3. 处理成绩分布数据...")
df_score = pd.read_csv(PATH_SCORE, sep='\t', encoding='utf-8-sig')
print(f"  原始形状: {df_score.shape}")

# 重命名列
df_score.columns = ['分数', '人数', '累计人数']

# 去掉备注行：只保留分数列完全是数字的行
df_score = df_score[df_score["分数"].astype(str).str.match(r"^\d+$")]

# 转换数据类型
df_score["分数"] = (
    df_score["分数"]
    .astype(str)
    .str.replace("分及以上", "", regex=False)
    .str.replace("分", "", regex=False)
    .astype(int)
)

for col in ["人数", "累计人数"]:
    df_score[col] = (
        df_score[col]
        .astype(str)
        .str.replace(r"[^\d]", "", regex=True)
        .astype(int)
    )

# 计算百分位
total = df_score["累计人数"].iloc[-1]
df_score["percentile"] = df_score["累计人数"] / total

# 添加年份列
df_score["年份"] = 2023

# 重排列及保存
df_score = df_score[["年份", "分数", "人数", "累计人数", "percentile"]]
df_score.to_csv(DATA_DIR / "2023年考生高考成绩分布表_clean.csv", index=False, encoding="utf-8-sig")

print(f"  清理后形状: {df_score.shape}")
print(f"  总人数: {total}")
print("  ✅ 已保存清理版本\n")

# ==================== 总结 ====================
print("🎉 所有数据处理完成！")
print(f"📊 df_combo 最终形状: {df_combo.shape}")
print(f"📊 df_major 最终形状: {df_major.shape}")
print(f"📊 df_score 最终形状: {df_score.shape}")
print(f"📁 清理文件保存在: {DATA_DIR}")

# %% [CELL 23]
# 7. 打印所有表的形状和预览
print("⇢ df_combo:", df_combo.shape, "行")
print(df_combo.head(3), "\n")

print("⇢ df_major:", df_major.shape, "行")
print(df_major.head(3), "\n")

print("⇢ df_score:", df_score.shape, "行")
print(df_score.head(3))

# %% [CELL 24]
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rcParams

# -------- 中文字体 --------
# 1. 指定黑体字体路径（Windows自带）
font_path = "C:/Windows/Fonts/simhei.ttf"
simhei_font = font_manager.FontProperties(fname=font_path)

# 2. 全局设置字体
plt.rcParams['font.family'] = simhei_font.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示



# %% [CELL 25]
import pandas as pd
from pathlib import Path

# 修正后的路径定义
BASE_DIR = Path("E:/代码/minicondapythonProject1")
DATA_DIR = BASE_DIR / "论文代码" / "志愿填报辅助系统" / "上海高考录取数据17-23年"

# 读取清理后的数据文件
df_score = pd.read_csv(DATA_DIR / "2023年考生高考成绩分布表_clean.csv", encoding="utf-8-sig")
df_major = pd.read_csv(DATA_DIR / "2023上海专业分数线_clean.csv", encoding="utf-8-sig")
df_combo = pd.read_csv(DATA_DIR / "subject_combo_to_mbti_clean.csv", encoding="utf-8-sig")

# 显示数据形状验证
print("读取清理数据完成！")
print(f"df_score 形状: {df_score.shape}")
print(f"df_major 形状: {df_major.shape}")
print(f"df_combo 形状: {df_combo.shape}")

# %% [CELL 26]
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# 确保已经定义了 DATA_DIR
BASE_DIR = Path("E:/代码/minicondapythonProject1")
DATA_DIR = BASE_DIR / "论文代码" / "志愿填报辅助系统" / "上海高考录取数据17-23年"

# 分数段分布（折线 + 累计人数）
plt.figure(figsize=(10,6))
sns.lineplot(x='分数', y='人数', data=df_score, label='各分数段人数')
sns.lineplot(x='分数', y='累计人数', data=df_score, label='累计人数')
plt.title('2023 上海考生分数段分布')
plt.xlabel('高考分数')
plt.ylabel('人数')
plt.legend()
plt.tight_layout()
plt.savefig(DATA_DIR / "score_line_2023.png", dpi=300)
plt.show()

# %% [CELL 27]
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
plt.savefig(DATA_DIR / "score_line_2023.png", dpi=300, bbox_inches='tight')
plt.show()

# %% [CELL 28]
#专业最低分箱线图（可看整体难度水平）
plt.figure(figsize=(12,6))
sns.boxplot(x='批次', y='最低分', data=df_major, palette='Set3')
plt.title('不同批次专业最低分分布')
plt.xlabel('批次')
plt.ylabel('最低分')
plt.tight_layout()
plt.savefig(DIR / "major_box_2023.png", dpi=300)  # ← 保存图片
plt.show()


# %% [CELL 29]
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
plt.savefig(DATA_DIR / "combo_mbti_bar_2023.png", dpi=300)
plt.show()

# %% [CELL 31]
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks


# %% [CELL 32]
from pathlib import Path
import pandas as pd

# 修正路径定义
BASE_DIR = Path("E:/代码/minicondapythonProject1")
DATA_DIR = BASE_DIR / "论文代码" / "志愿填报辅助系统" / "上海高考录取数据17-23年"

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

# %% [CELL 33]
import pandas as pd
import numpy as np
from pathlib import Path

# 1. 数据加载
BASE_DIR = Path("E:/代码/minicondapythonProject1")
DATA_DIR = BASE_DIR / "论文代码" / "志愿填报辅助系统" / "上海高考录取数据17-23年"

# 加载专业分数线数据
df_major = pd.read_csv(DATA_DIR / "2023上海专业分数线_clean.csv", encoding="utf-8-sig")
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

# %% [CELL 34]
import pandas as pd
import numpy as np
from pathlib import Path

# 1. 数据加载
BASE_DIR = Path("E:/代码/minicondapythonProject1")
DATA_DIR = BASE_DIR / "论文代码" / "志愿填报辅助系统" / "上海高考录取数据17-23年"

# 加载成绩分布数据
df_score = pd.read_csv(DATA_DIR / "2023年考生高考成绩分布表_clean.csv", encoding="utf-8-sig")
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

# %% [CELL 35]
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

# %% [CELL 36]
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
BASE_DIR = Path("E:/代码/minicondapythonProject1")
DATA_DIR = BASE_DIR / "论文代码" / "志愿填报辅助系统" / "上海高考录取数据17-23年"
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
plt.show()

# 9. 显示训练历史（如果可用）
if hasattr(mlp_model, 'loss_curve_'):
    plt.figure(figsize=(10, 4))
    plt.plot(mlp_model.loss_curve_)
    plt.title('MLP 训练损失曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "training_loss.png", dpi=300)
    plt.show()

# %% [CELL 37]
# 2. 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %% [CELL 38]
# 3. 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# %% [CELL 39]
# 4. 重塑为 LSTM 要求的 3D 张量 (samples, timesteps, features)
#    这里直接用 timesteps=1
X_train_scaled = X_train_scaled[:, None, :]
X_test_scaled  = X_test_scaled[:,  None, :]

# %% [CELL 40]
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


# %% [CELL 41]
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

# %% [CELL 42]
# 8. 保存模型和 scaler
MODEL_DIR = DIR / "lstm_rank_model"
MODEL_DIR.mkdir(exist_ok=True)
model.save(MODEL_DIR / "lstm_rank.h5")
pd.to_pickle(scaler, MODEL_DIR / "scaler_rank.pkl")
print(f"✔ 模型和 scaler 已保存到：{MODEL_DIR}")

# %% [CELL 43]
#可视化
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rcParams

# -------- 中文字体 --------
# 1. 指定黑体字体路径（Windows自带）
font_path = "C:/Windows/Fonts/simhei.ttf"
simhei_font = font_manager.FontProperties(fname=font_path)

# 2. 全局设置字体
plt.rcParams['font.family'] = simhei_font.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# %% [CELL 44]
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
plt.show()


# %% [CELL 45]
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
plt.show()

# %% [CELL 46]
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
plt.show()

# %% [CELL 47]
from sklearn.metrics import r2_score

# —— 假设前面已经计算好了 y_pred, 并且 y_test 可用 —— 

# 4. 计算并打印 R-squared（R方值）
r_squared = r2_score(y_test, y_pred)
print(f"R-squared (R方) = {r_squared:.2f}")

print("\n✓ 数据可视化内容已添加完成。")


# %% [CELL 48]
from sklearn.metrics import mean_absolute_percentage_error

# 残差直方图
residuals = y_test - y_pred.ravel()
plt.figure(figsize=(6,4))
plt.hist(residuals, bins=30, edgecolor='k')
plt.title('预测残差分布')
plt.xlabel('实际 - 预测')
plt.ylabel('样本数')
plt.tight_layout()
plt.show()

# MAPE
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE = {mape:.2%}")


# %% [CELL 50]
import pandas as pd
from pathlib import Path

# 1. 路径定义 - 修正为正确的路径
BASE_DIR = Path("E:/代码/minicondapythonProject1")
DATA_DIR = BASE_DIR / "论文代码" / "志愿填报辅助系统" / "上海高考录取数据17-23年"
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

# %% [CELL 51]
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

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

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
model = LSTMModel(input_size=n_feat, hidden_size=64, output_size=1)

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
            optimizer.zero_grad()
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
    model.load_state_dict(torch.load('best_model.pth'))
    return train_losses, val_losses

# 训练模型
print("开始训练 LSTM 模型...")
train_losses, val_losses = train_with_early_stopping(model, train_loader, test_loader)

# 8. 评估
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).squeeze().numpy()

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

# %% [CELL 52]
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks

# %% [CELL 53]
# 0. 路径
DIR         = Path(r"C:\Users\Lenovo\Desktop\志愿填报辅助系统\上海高考录取数据17-23年\用到的")
PATH_RANK   = DIR / "上海一分一段_2017-2022_clean.csv"
PATH_MAJOR  = DIR / "2023上海专业分数线_model_ready.csv"


# %% [CELL 54]
# 1. 读取数据
df_rank  = pd.read_csv(PATH_RANK,  encoding="utf-8-sig")
df_major = pd.read_csv(PATH_MAJOR, encoding="utf-8-sig")


# %% [CELL 55]
# 2. 确保 df_major 只有 2023 年的那一行（如果 clean 表里还有其它年份可以先筛）
df_major_2023 = df_major.copy()  # 已经是 2023 年数据

# %% [CELL 56]
# 3. 把专业表的数值特征“广播”到所有分数行
#    直接将那一行的特征附到 df_rank
#    （无需 on='年份'，因为始终用 2023 年数据）
for col in ["批次_code", "最低分", "最低位次", "平均分"]:
    df_rank[col] = df_major_2023[col].iloc[0]

# %% [CELL 57]
# 4. 构造衍生特征
total_n = df_rank["累计人数"].max()
df_rank["score_gap"] = df_rank["分数"] - df_rank["最低分"]
df_rank["rank_abs"]  = (df_rank["percentile"] * total_n).astype(int)
df_rank["rank_gap"]  = df_rank["rank_abs"] - df_rank["最低位次"]

# %% [CELL 58]
# 5. 选择特征和目标
FEATURE_COLS = [
    "分数", "本段人数", "累计人数",
    "批次_code", "最低分", "最低位次", "平均分",
    "score_gap", "rank_gap"
]
TARGET_COL = "percentile"

X = df_rank[FEATURE_COLS].astype("float32").values
y = df_rank[TARGET_COL].astype("float32").values


# %% [CELL 59]
# 6. 划分、标准化、reshape
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)[:, None, :]
X_test_s  = scaler.transform(X_test)[:,  None, :]


# %% [CELL 60]
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


# %% [CELL 61]
from sklearn.metrics import mean_absolute_error, r2_score

# 8. 评估
y_pred = model.predict(X_test_s)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"Test MAE: {mae:.4f}")
print(f"R²      : {r2:.4f}")

# %% [CELL 62]
# 9. 保存模型与 scaler
MODEL_DIR = DIR / "lstm_rank_opt"
MODEL_DIR.mkdir(exist_ok=True)
model.save(MODEL_DIR / "lstm_rank_opt.h5")
pd.to_pickle(scaler, MODEL_DIR / "scaler_rank_opt.pkl")
print("优化后的模型与 scaler 已保存至：", MODEL_DIR)

# %% [CELL 63]
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


# %% [CELL 65]
from pathlib import Path

# 把下面这一行替换成你本地的完整目录
DIR = Path(r"C:\Users\Lenovo\Desktop\志愿填报辅助系统\上海高考录取数据17-23年\用到的")

# 验证一下目录里到底有哪些文件
print(list(DIR.iterdir()))


# %% [CELL 66]
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. 设定正确的目录 - 修正路径
BASE_DIR = Path("E:/代码/minicondapythonProject1")
DATA_DIR = BASE_DIR / "论文代码" / "志愿填报辅助系统" / "上海高考录取数据17-23年"

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

# %% [CELL 67]
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. 设定正确的目录
DIR = Path(r"C:\Users\Lenovo\Desktop\志愿填报辅助系统\上海高考录取数据17-23年\用到的")

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


# %% [CELL 68]
from pathlib import Path
import pandas as pd

# 1. 把 DIR 改成你本地的完整目录
DIR = Path(r"C:\Users\Lenovo\Desktop\志愿填报辅助系统\上海高考录取数据17-23年\用到的")

# 2. 读取清洗后的分数分布表
df_rank = pd.read_csv(DIR / "上海一分一段_2017-2022_clean.csv", encoding="utf-8-sig")

# 3. 读取 2023 专业分数线特征
df_major = pd.read_csv(DIR / "2023上海专业分数线_model_ready.csv", encoding="utf-8-sig")

# 确认都加载成功
print(df_rank.shape, df_major.shape)


# %% [CELL 69]
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


# %% [CELL 71]
import pandas as pd
from pathlib import Path
import re

# 0. 路径
BASE_DIR = Path("E:/代码/minicondapythonProject1")
DATA_DIR = BASE_DIR / "论文代码" / "志愿填报辅助系统" / "上海高考录取数据17-23年"
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

# %% [CELL 73]
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

print("开始构建推荐系统（使用 scikit-learn SVD）...")

# ---------- 1. 路径 ----------
BASE_DIR = Path("E:/代码/minicondapythonProject1")
DATA_DIR = BASE_DIR / "论文代码" / "志愿填报辅助系统" / "上海高考录取数据17-23年"

# ---------- 2. 读取数据 ----------
# 读取选科组合数据
df_combo = pd.read_csv(DATA_DIR / "subject_combo_to_mbti_clean.csv", encoding="utf-8-sig")
df_combo = df_combo.rename(columns={"subject_combo": "combo", "count": "count"})

# 读取专业MBTI数据
df_items = pd.read_csv(DATA_DIR / "2023上海专业分数线_with_PredictedMBTI.csv", encoding="utf-8-sig")
df_items["school_major"] = df_items["院校名称"] + "_" + df_items["专业名称"]

print(f"选科组合数据形状: {df_combo.shape}")
print(f"专业数据形状: {df_items.shape}")

# ---------- 3. 学生(pseudo-user) 画像 ----------
df_users = df_combo[["combo", "mbti", "count"]].copy()
df_users["uid"] = "u_" + df_users.index.astype(str)

print(f"用户数据形状: {df_users.shape}")
print(f"用户MBTI分布:\n{df_users['mbti'].value_counts()}")

# ---------- 4. 构造 MBTI 匹配度函数 ----------
def mbti_weight(stu_mbti, maj_mbti):
    if stu_mbti == maj_mbti:           # 完全一致
        return 1.0
    if stu_mbti[:2] == maj_mbti[:2]:   # 中间两位相同（NT/NF/ST/SF）
        return 0.5
    return 0.0

# ---------- 5. 生成伪评分表 ----------
# 创建用户-项目评分矩阵
user_ids = df_users["uid"].tolist()
item_ids = df_items["school_major"].tolist()

# 创建空的评分矩阵
rating_matrix = np.zeros((len(user_ids), len(item_ids)))

# 批次难度映射
batch_difficulty = {
    "专科批": 1,
    "本科批": 3,
    "本科提前批": 4,
    "高职提前批": 2,
    "提前批": 4,
    "艺术类本科批": 3,
    "体育类本科批": 3,
}

print("批次类别:", df_items["批次"].unique())

# 为每个用户-项目对计算评分
for i, user_row in df_users.iterrows():
    for j, item_row in df_items.iterrows():
        # 计算难度
        difficulty = batch_difficulty.get(item_row["批次"], 3)  # 默认3
        difficulty = difficulty / max(batch_difficulty.values())  # 归一化

        # 计算MBTI权重
        mbti_w = mbti_weight(user_row["mbti"], item_row["Predicted_MBTI"])

        # 计算原始评分
        raw_score = mbti_w * (1 - difficulty)
        rating = (raw_score * 4 + 1)  # 缩放到1-5范围

        rating_matrix[i, j] = rating

print(f"评分矩阵形状: {rating_matrix.shape}")
print(f"评分范围: {rating_matrix.min():.2f} - {rating_matrix.max():.2f}")

# ---------- 6. 使用 TruncatedSVD 进行矩阵分解 ----------
n_components = min(50, min(rating_matrix.shape) - 1)  # 确保不超过矩阵维度
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_factors = svd.fit_transform(rating_matrix)
item_factors = svd.components_.T

print(f"SVD分解完成: {n_components} 个成分")
print(f"解释方差比: {svd.explained_variance_ratio_.sum():.3f}")

# 重建评分矩阵
reconstructed_ratings = user_factors @ item_factors.T

print(f"重建评分矩阵形状: {reconstructed_ratings.shape}")

# ---------- 7. Top-N 推荐函数 ----------
def recommend(user_index, N=10):
    user_scores = reconstructed_ratings[user_index]
    top_indices = np.argsort(user_scores)[::-1][:N]
    return [(item_ids[j], user_scores[j]) for j in top_indices]

# ---------- 8. 批量输出每个选科组合的 Top-10 ----------
all_recs = []
for i, uid in enumerate(user_ids):
    top10 = recommend(i, 10)
    for rank, (itm, score) in enumerate(top10, 1):
        all_recs.append([uid, rank, itm, score])

out = pd.DataFrame(all_recs, columns=["uid", "rank", "school_major", "est_score"])

# 添加用户信息和专业信息
out = out.merge(df_users[["uid", "combo", "mbti"]], on="uid")
out = out.merge(df_items[["school_major", "院校名称", "专业名称", "最低分", "Predicted_MBTI"]], on="school_major")

# 重新排列列顺序
out = out[["uid", "combo", "mbti", "rank", "院校名称", "专业名称", "最低分", "Predicted_MBTI", "est_score"]]

# 保存结果
out.to_csv(DATA_DIR / "combo_based_recommendations_sklearn.csv", index=False, encoding="utf-8-sig")
print("✅ 推荐结果已保存 → combo_based_recommendations_sklearn.csv")

# ---------- 9. 查看示例 ----------
print("\n📋 推荐结果示例 (前20行):")
print(out.head(20))

# ---------- 10. 可视化分析 ----------
plt.figure(figsize=(15, 10))

# 原始评分分布
plt.subplot(2, 3, 1)
sns.histplot(rating_matrix.flatten(), bins=20, kde=True)
plt.title('原始评分分布')
plt.xlabel('评分')
plt.ylabel('数量')

# 重建评分分布
plt.subplot(2, 3, 2)
sns.histplot(reconstructed_ratings.flatten(), bins=20, kde=True)
plt.title('重建评分分布')
plt.xlabel('重建评分')
plt.ylabel('数量')

# 最终推荐分数分布
plt.subplot(2, 3, 3)
sns.histplot(out['est_score'], bins=20, kde=True)
plt.title('最终推荐分数分布')
plt.xlabel('推荐分数')
plt.ylabel('推荐数量')

# 按选科组合的平均分数
plt.subplot(2, 3, 4)
combo_scores = out.groupby('combo')['est_score'].mean().sort_values(ascending=False)
combo_scores.plot(kind='bar', color='skyblue')
plt.title('各选科组合平均推荐分数')
plt.xlabel('选科组合')
plt.ylabel('平均推荐分数')
plt.xticks(rotation=45)

# MBTI匹配分析
plt.subplot(2, 3, 5)
mbti_match = out.groupby(['mbti', 'Predicted_MBTI']).size().reset_index(name='count')
top_matches = mbti_match.nlargest(10, 'count')
sns.barplot(data=top_matches, x='count', y='mbti', hue='Predicted_MBTI')
plt.title('Top MBTI匹配组合')
plt.xlabel('匹配数量')

plt.tight_layout()
plt.savefig(DATA_DIR / "recommendation_analysis_sklearn.png", dpi=300)
plt.show()

# ---------- 11. 详细统计 ----------
print(f"\n🎯 推荐系统统计:")
print(f"用户数量: {len(user_ids)}")
print(f"专业数量: {len(item_ids)}")
print(f"总推荐数量: {len(out)}")
print(f"平均推荐分数: {out['est_score'].mean():.3f} ± {out['est_score'].std():.3f}")

print(f"\n📊 按选科组合的推荐统计:")
combo_stats = out.groupby('combo').agg({
    'est_score': ['mean', 'std', 'count'],
    '最低分': 'mean'
}).round(3)
combo_stats.columns = ['平均分数', '分数标准差', '推荐数量', '平均最低分']
print(combo_stats)

print(f"\n🎓 热门推荐专业 (前10):")
top_majors = out['专业名称'].value_counts().head(10)
print(top_majors)

# ---------- 12. 保存分析报告 ----------
analysis_report = DATA_DIR / "recommendation_analysis_report_sklearn.txt"
with open(analysis_report, 'w', encoding='utf-8') as f:
    f.write("=== 推荐系统分析报告 (scikit-learn SVD) ===\n\n")
    f.write(f"用户数量: {len(user_ids)}\n")
    f.write(f"专业数量: {len(item_ids)}\n")
    f.write(f"总推荐数量: {len(out)}\n")
    f.write(f"平均推荐分数: {out['est_score'].mean():.3f}\n")
    f.write(f"SVD成分数: {n_components}\n")
    f.write(f"解释方差比: {svd.explained_variance_ratio_.sum():.3f}\n")
    f.write(f"\n选科组合统计:\n")
    f.write(combo_stats.to_string())
    f.write(f"\n\n热门推荐专业:\n")
    f.write(top_majors.to_string())

print(f"✅ 分析报告已保存: {analysis_report}")

# %% [CELL 75]
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
model = InterpretableNet(input_size)
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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ---------- 3. 训练模型 ----------
def train_model(model, train_loader, epochs=100):
    model.train()
    train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
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
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.squeeze().numpy()

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
    X_tensor.requires_grad_(True)

    # 前向传播
    outputs = model(X_tensor)

    # 计算梯度
    gradients = []
    for i in range(outputs.shape[0]):
        model.zero_grad()
        outputs[i].backward(retain_graph=True)
        grad = X_tensor.grad[i].abs().numpy()
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

    for value in feature_values:
        X_temp = X_base.clone()
        X_temp[:, feature_index] = value
        with torch.no_grad():
            predictions = model(X_temp)
        pdp_values.append(predictions.mean().item())

    return pdp_values

# 为每个特征计算PDP
feature_values_dict = {}
for i, feature in enumerate(FEATURES):
    feature_min = X_test_tensor[:, i].min().item()
    feature_max = X_test_tensor[:, i].max().item()
    feature_values = torch.linspace(feature_min, feature_max, 50)
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
    X_sample = X_tensor[sample_index:sample_index+1]
    X_sample.requires_grad_(True)

    # 预测
    with torch.no_grad():
        prediction = model(X_sample).item()

    # 梯度
    model.zero_grad()
    output = model(X_sample)
    output.backward()
    gradients = X_sample.grad[0].abs().numpy()

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
mbti_original = feature_values_dict['mbti_w'] * X_std[0] + X_mean[0]
plt.plot(mbti_original, pdp_results['mbti_w'], linewidth=3, color='#FF6B6B')
plt.xlabel('MBTI 匹配度 (原始尺度)')
plt.ylabel('平均预测评分')
plt.title('MBTI匹配度部分依赖图', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# 6.4 部分依赖图 - 难度
plt.subplot(2, 3, 4)
difficulty_original = feature_values_dict['difficulty'] * X_std[1] + X_mean[1]
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

# %% [CELL 76]
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

    # 构造候选数据
    cand_list = []
    for _, user_row in df_users.iterrows():
        for _, item_row in df_items.iterrows():
            # MBTI 匹配度函数
            def mbti_weight(stu_mbti, maj_mbti):
                if stu_mbti == maj_mbti:
                    return 1.0
                if stu_mbti[:2] == maj_mbti[:2]:
                    return 0.5
                return 0.0

            # 批次难度
            batch_difficulty = {
                "专科批": 1, "本科批": 3, "本科提前批": 4,
                "高职提前批": 2, "提前批": 4, "艺术类本科批": 3, "体育类本科批": 3
            }

            difficulty = batch_difficulty.get(item_row["批次"], 3)
            difficulty = difficulty / max(batch_difficulty.values())

            mbti_w = mbti_weight(user_row["mbti"], item_row["Predicted_MBTI"])
            raw_score = mbti_w * (1 - difficulty)
            rating = (raw_score * 4 + 1)

            cand_list.append({
                "uid": user_row["uid"],
                "combo": user_row["combo"],
                "mbti": user_row["mbti"],
                "school_major": item_row["school_major"],
                "mbti_w": mbti_w,
                "difficulty": difficulty,
                "rating": rating,
                "院校名称": item_row["院校名称"],
                "专业名称": item_row["专业名称"],
                "Predicted_MBTI": item_row["Predicted_MBTI"],
                "最低分": item_row["最低分"]
            })

    cand = pd.DataFrame(cand_list)
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

# %% [CELL 77]
import pandas as pd
import numpy as np

# 假设 cand 已经在前面构造好了，并包含列：mbti_w, difficulty, rating
# 例如：
# cand = pd.read_csv("combo_based_recommendations_input.csv")

from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt

# ---------- 1. 特征和目标 ----------
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




# %% [CELL 78]
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


# %% [CELL 79]
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
save_dir = r"C:\Users\Lenovo\Desktop\志愿填报辅助系统\上海高考录取数据17-23年\用到的"
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


# %% [CELL 80]
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
BASE_DIR = Path("E:/代码/minicondapythonProject1")
DATA_DIR = BASE_DIR / "论文代码" / "志愿填报辅助系统" / "上海高考录取数据17-23年"

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
cand_list = []

for _, user_row in df_users.iterrows():
    for _, item_row in df_items.iterrows():
        # 批次难度映射
        batch_difficulty = {
            "专科批": 1, "本科批": 3, "本科提前批": 4,
            "高职提前批": 2, "提前批": 4, "艺术类本科批": 3, "体育类本科批": 3
        }

        difficulty = batch_difficulty.get(item_row["批次"], 3)
        difficulty = difficulty / max(batch_difficulty.values())

        mbti_w = mbti_weight(user_row["mbti"], item_row["Predicted_MBTI"])
        raw_score = mbti_w * (1 - difficulty)
        rating = (raw_score * 4 + 1)

        cand_list.append({
            "uid": user_row["uid"],
            "combo": user_row["combo"],
            "mbti": user_row["mbti"],
            "school_major": item_row["school_major"],
            "mbti_w": mbti_w,
            "difficulty": difficulty,
            "rating": rating,
            "院校名称": item_row["院校名称"],
            "专业名称": item_row["专业名称"],
            "Predicted_MBTI": item_row["Predicted_MBTI"],
            "最低分": item_row["最低分"]
        })

cand = pd.DataFrame(cand_list)
print(f"候选数据形状: {cand.shape}")

# ---------- 6. 矩阵分解推荐 ----------
print("开始矩阵分解推荐...")

# 创建用户-项目评分矩阵
user_ids = df_users["uid"].tolist()
item_ids = df_items["school_major"].tolist()

rating_matrix = np.zeros((len(user_ids), len(item_ids)))

for i, user_row in df_users.iterrows():
    for j, item_row in df_items.iterrows():
        difficulty = batch_difficulty.get(item_row["批次"], 3)
        difficulty = difficulty / max(batch_difficulty.values())
        mbti_w = mbti_weight(user_row["mbti"], item_row["Predicted_MBTI"])
        raw_score = mbti_w * (1 - difficulty)
        rating = (raw_score * 4 + 1)
        rating_matrix[i, j] = rating

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

all_recs = []
for i, uid in enumerate(user_ids):
    top10 = recommend(i, 10)
    for rank, (itm, score) in enumerate(top10, 1):
        all_recs.append([uid, rank, itm, score])

out = pd.DataFrame(all_recs, columns=["uid", "rank", "school_major", "est_score"])

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

# %% [CELL 81]
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体（防止中文乱码）
plt.rcParams['font.family'] = 'SimHei'  # Windows 系统默认黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ---------- 1. 路径设置 ----------
BASE_DIR = Path("E:/代码/minicondapythonProject1")
DATA_DIR = BASE_DIR / "论文代码" / "志愿填报辅助系统" / "上海高考录取数据17-23年"

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

# %% [CELL 83]
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
output_path = "lstm_residuals_hist.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"图像已保存至：{output_path}")



# %% [CELL 84]
print("X_train_scaled_2d:", X_train_scaled_2d.shape)
print("y_train:",            y_train.shape)


# %% [CELL 85]
# —— 重新切分并标准化 (为所有模型共用)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %% [CELL 86]
# 特征 & 目标
X_tab = df_rank[FEATURE_COLS].values.astype("float32")    # shape=(N,1)
y_tab = df_rank[TARGET_COL].values.astype("float32")      # shape=(N,)

# %% [CELL 87]
# 切分
X_train_tab, X_test_tab, y_train_tab, y_test_tab = train_test_split(
    X_tab, y_tab, test_size=0.2, random_state=42
)

# %% [CELL 88]
# 标准化（只对需要的模型，如 Linear/SVR）
scaler_ml = StandardScaler()
X_train_tab_scaled = scaler_ml.fit_transform(X_train_tab)  # shape=(n_train,1)
X_test_tab_scaled  = scaler_ml.transform(X_test_tab)       # shape=(n_test,1)


# %% [CELL 89]
# 为 LSTM/GRU 保留 3D 张量版本
X_train_scaled = X_train_tab_scaled[:, None, :]            # shape=(n_train,1,1)
X_test_scaled  = X_test_tab_scaled[:,  None, :]            # shape=(n_test,1,1)


# %% [CELL 90]
# 为传统机器学习模型准备 2D 版本
X_train_scaled_2d = X_train_tab_scaled.squeeze(axis=1)     # shape=(n_train,)
X_test_scaled_2d  = X_test_tab_scaled.squeeze(axis=1)      # shape=(n_test,)

# %% [CELL 91]
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


# %% [CELL 92]
#补充多模型对比
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings, sys
warnings.filterwarnings("ignore")

# %% [CELL 93]
# 检测 xgboost
try:
    from xgboost import XGBRegressor
    has_xgb = True
except ImportError:
    has_xgb = False
    print("⚠ 未检测到 xgboost，已跳过 XGBRegressor", file=sys.stderr)


# %% [CELL 94]
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

# %% [CELL 95]
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


# %% [CELL 96]
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


# %% [CELL 97]
#LSTM 预测并加入对比
y_pred_lstm = model.predict(X_test_scaled, verbose=0).squeeze()
results.append(("LSTM_64",
                mean_absolute_error(y_test_tab, y_pred_lstm),
                r2_score(y_test_tab, y_pred_lstm)))


# %% [CELL 98]
import pandas as pd

df_res = pd.DataFrame(results, columns=["Model", "MAE", "R2"]).sort_values("MAE")
print("\n===== Rank-Prediction Benchmark =====")
print(df_res.to_string(index=False, formatters={"MAE": "{:.4f}".format,
                                               "R2": "{:.3f}".format}))

# 保存 CSV
MODEL_DIR = DIR / "benchmark_models"
MODEL_DIR.mkdir(exist_ok=True)
df_res.to_csv(MODEL_DIR / "benchmark_rank_models.csv",
              index=False, encoding="utf-8-sig")


# %% [CELL 99]


