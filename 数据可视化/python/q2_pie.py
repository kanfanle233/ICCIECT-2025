"""
饼图展示：商家A各图书分类的销量占比。

教学重点：
  - 使用 matplotlib 绘制饼图（Pie Chart）
  - autopct 参数控制百分比标签的显示格式
  - startangle 控制饼图起始角度
  - axis("equal") 保证饼图为正圆形
"""

# --- 1. 导入与后端设置 ---
import matplotlib

matplotlib.use("Agg")  # 使用非交互式后端，适合脚本环境保存图片

import matplotlib.pyplot as plt

# --- 2. 中文字体与负号显示配置 ---
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # 设置中文字体为微软雅黑
plt.rcParams["axes.unicode_minus"] = False  # 解决负号 '-' 显示为方块的问题

# --- 3. 数据准备 ---
# 直接改这里的数据即可
labels = ["哲学", "历史", "教育", "科技", "文学", "经济"]  # 图书分类标签
merchant_a = [25, 20, 36, 40, 75, 90]  # 商家A各分类销量

# --- 4. 绘制饼图 ---
plt.figure(figsize=(6, 6))  # 创建正方形画布，保证饼图不变形
plt.pie(merchant_a, labels=labels, autopct="%1.1f%%", startangle=90)
# autopct: 显示百分比，保留一位小数；startangle: 第一片从 90 度位置开始
plt.title("202314109方昕哲 图书销售量")
plt.axis("equal")  # 确保饼图绘制为正圆而非椭圆
plt.tight_layout()  # 自动调整布局，避免标签被裁剪
plt.savefig("q2_pie.png", dpi=200)  # 保存为 PNG 图片，分辨率 200 dpi
