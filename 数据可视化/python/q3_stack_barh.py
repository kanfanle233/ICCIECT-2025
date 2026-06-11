"""
堆叠水平柱状图：各班级男女生人数对比。

教学重点：
  - 使用 matplotlib 绘制水平堆叠柱状图（Stacked Horizontal Bar Chart）
  - barh 函数的 left 参数实现堆叠效果
  - yticks 替换刻度标签为班级名称
"""

# --- 1. 导入与后端设置 ---
import matplotlib

matplotlib.use("Agg")  # 使用非交互式后端，适合脚本环境保存图片

import matplotlib.pyplot as plt

# --- 2. 中文字体与负号显示配置 ---
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # 设置中文字体为微软雅黑
plt.rcParams["axes.unicode_minus"] = False  # 解决负号 '-' 显示为方块的问题

# --- 3. 数据准备 ---
# 直接改这里的数据
classes = ["1班", "2班", "3班", "4班"]  # 班级名称
boys = [5, 7, 3, 6]   # 各班男生人数
girls = [3, 4, 7, 2]   # 各班女生人数

y = range(len(classes))  # 每个班级在 y 轴上的位置索引

# --- 4. 绘制堆叠水平柱状图 ---
plt.figure(figsize=(7, 4))  # 创建画布，宽 7 英寸、高 4 英寸
plt.barh(y, boys, label="男生")               # 先画男生柱子
plt.barh(y, girls, left=boys, label="女生")    # 女生柱子从男生柱子右端开始，形成堆叠
plt.yticks(y, classes)  # 将 y 轴刻度替换为班级名称
plt.ylabel("班级")
plt.xlabel("人数")
plt.title("202314109方昕哲 男女人数对比")
plt.legend()  # 显示图例
plt.tight_layout()  # 自动调整布局，避免标签被裁剪
plt.savefig("q3_stack_barh.png", dpi=200)  # 保存为 PNG 图片，分辨率 200 dpi
