import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "PingFang SC", "Arial Unicode MS"
]
plt.rcParams["axes.unicode_minus"] = False

types = ["哲学", "历史", "教育", "科技", "文学", "经济"]
A = [25, 20, 36, 40, 75, 90]

plt.pie(A, labels=types, autopct="%.1f%%")
plt.title("202314109方昕哲图书销售量")

plt.show()