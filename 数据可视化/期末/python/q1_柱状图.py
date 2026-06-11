import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "PingFang SC", "Arial Unicode MS"
]
plt.rcParams["axes.unicode_minus"] = False

types = ["哲学", "历史", "教育", "科技", "文学", "经济"]
A = [25, 20, 36, 40, 75, 90]
B = [35, 26, 45, 50, 35, 66]

x = list(range(len(types)))
w = 0.35

plt.bar([i - w / 2 for i in x], A, width=w, label="商家A")
plt.bar([i + w / 2 for i in x], B, width=w, label="商家B")

plt.xticks(x, types)
plt.xlabel("图书类别")
plt.ylabel("销售量")
plt.title("202314109方昕哲图书销售量")
plt.legend()

plt.show()