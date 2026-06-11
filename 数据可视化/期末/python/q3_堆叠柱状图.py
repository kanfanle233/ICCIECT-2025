import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "PingFang SC", "Arial Unicode MS"
]
plt.rcParams["axes.unicode_minus"] = False

classes = ["1班", "2班", "3班", "4班"]
boys = [5, 7, 3, 6]
girls = [3, 4, 7, 2]

x = list(range(len(classes)))

plt.bar(x, boys, label="男生")
plt.bar(x, girls, bottom=boys, label="女生")

plt.xticks(x, classes)
plt.xlabel("班级")
plt.ylabel("人数")
plt.title("202314109方昕哲绘制男女人数对比")
plt.legend()

plt.show()