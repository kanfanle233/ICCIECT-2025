# -*- coding: utf-8 -*-
"""
File: 001_数据读取与基础可视化.py
Purpose: 初始读取、MBTI分布与基础图表
Source notebook: 高考数据处理大修版本-checkpoint.ipynb
Execution order: keep numeric order 001 -> 010
"""

# ===== From Notebook CELL 1 =====
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 统一的工程/数据目录，避免硬编码绝对路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from gaokao_recommender.paths import *

DATA_DIR = DATA_RAW_DIR
OUTPUT_DIR = DATA_PROCESSED_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== From Notebook CELL 2 =====
# 正确定义文件路径
PATH_COMBO_MBTI = DATA_DIR / "subject_combo_to_mbti.txt"
PATH_MAJOR = DATA_DIR / "2023上海专业分数线.txt"
PATH_SCORE_DIST = DATA_DIR / "2023年考生高考成绩分布表（上海市）.txt"

# 读取数据
df_combo = pd.read_csv(PATH_COMBO_MBTI, sep='\t', encoding='utf-8-sig')
df_major = pd.read_csv(PATH_MAJOR, sep='\t', encoding='utf-8-sig')
df_score = pd.read_csv(PATH_SCORE_DIST, sep='\t', encoding='utf-8-sig')



# ===== From Notebook CELL 3 =====
# 3. 对应列名（如有不同请检查 txt 头部改这里）
df_combo = df_combo.rename(columns={
    "mbti": "MBTI",
    "count": "人数count",
    "subject_combo": "选科组合"
})



# ===== From Notebook CELL 4 =====
# 4. 按 MBTI 累加总人数，得到全体分布
mbti_pie = df_combo.groupby("MBTI")["人数count"].sum().sort_values(ascending=False)



# ===== From Notebook CELL 5 =====
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
OUT_FIG = OUTPUT_DIR / "fig_mbti_pie_2023.png"
plt.savefig(OUT_FIG, dpi=300)
plt.show()

print(f"饼图已保存：{OUT_FIG}")



# ===== From Notebook CELL 6 =====
# 第6步中的DIR变量需要定义
# 建议添加：
import pathlib
DIR = OUTPUT_DIR
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
OUT_DIR = OUTPUT_DIR
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





