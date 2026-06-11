"""
教学示例：numpy matplot002

- 功能：演示 数据可视化 中与“numpy matplot002”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""



import matplotlib.pyplot as plt

# 数据
v_1 = [20, 15, 18, 16, 21, 14, 10]  # 最高气温
v_2 = [12, 8, 14, 10, 13, 9, 4]  # 最低气温
x = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# 解决中文乱码
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 创建画布：2 行 2 列子图
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 扁平化子图数组，方便循环
axes = axes.flatten()

for i, ax in enumerate(axes):
    ax.plot(x, v_1, color="red", marker="o", label="最高气温")
    ax.plot(x, v_2, color="blue", marker="o", label="最低气温")
    ax.set_title(f"第{i + 1}组一周气温")
    ax.set_xlabel("日期")
    ax.set_ylabel("气温(℃)")
    ax.legend()
    ax.grid(True, alpha=0.3)

# 自动调整子图间距，防止重叠
plt.tight_layout()
plt.show()