"""
柱状图对比：不同商家的图书分类销量。

教学重点：
  - 使用 matplotlib 绘制分组柱状图（Grouped Bar Chart）
  - 通过偏移 x 坐标实现两组柱子并排显示
  - 中文字体配置与图例设置
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
merchant_b = [35, 26, 45, 50, 35, 66]  # 商家B各分类销量

x = range(len(labels))  # 每个分类的 x 位置索引
width = 0.35  # 每根柱子的宽度

# --- 4. 绘制分组柱状图 ---
plt.figure(figsize=(8, 4))  # 创建画布，宽 8 英寸、高 4 英寸
plt.bar([i - width / 2 for i in x], merchant_a, width=width, label="商家A")  # 商家A的柱子左移半个宽度
plt.bar([i + width / 2 for i in x], merchant_b, width=width, label="商家B")  # 商家B的柱子右移半个宽度
plt.xticks(list(x), labels)  # 将 x 轴刻度替换为分类名称
plt.ylabel("销量")
plt.title("202314109方昕哲 图书销售量")
plt.legend()  # 显示图例
plt.tight_layout()  # 自动调整布局，避免标签被裁剪
plt.savefig("q1_bar.png", dpi=200)  # 保存为 PNG 图片，分辨率 200 dpi
