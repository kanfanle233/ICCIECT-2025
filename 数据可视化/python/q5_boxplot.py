"""
箱线图展示：不同类别的数据分布与离群值检测。

教学重点：
  - 使用 seaborn 绘制箱线图（Box Plot）
  - 箱线图展示中位数、四分位距和异常值
  - 结合 pandas DataFrame 使用 seaborn 的数据接口
"""

# --- 1. 导入与后端设置 ---
import matplotlib

matplotlib.use("Agg")  # 使用非交互式后端，适合脚本环境保存图片

import matplotlib.pyplot as plt
import pandas as pd    # 数据处理库，用于构建 DataFrame
import seaborn as sns  # 基于 matplotlib 的统计可视化库

# --- 2. 数据准备 ---
data = {
    "Category": ["A", "A", "B", "B", "C", "C"],  # 类别列
    "Value": [10, 15, 20, 25, 30, 35],            # 数值列
}

# --- 3. 绘制箱线图 ---
sns.boxplot(data=pd.DataFrame(data), x="Category", y="Value")
# x: 分类变量（决定箱线图的分组）
# y: 数值变量（决定每个箱线图的分布范围）
plt.tight_layout()  # 自动调整布局，避免标签被裁剪
plt.savefig("q5_boxplot.png", dpi=200)  # 保存为 PNG 图片，分辨率 200 dpi
