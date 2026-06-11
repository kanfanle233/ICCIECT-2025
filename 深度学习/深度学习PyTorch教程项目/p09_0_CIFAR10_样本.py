"""
加载 CIFAR-10 数据集并可视化样本图像及其中文类别标签。
教学重点：torchvision.datasets.CIFAR10 数据集加载、subplot 批量展示、中文字体配置。
"""

# p090_CIFAR_样本.py
from torchvision import datasets as ds
import matplotlib.pyplot as ppl

# CIFAR-10 的10个类别名称（中文）
CLASS = ('飞机', '汽车', '鸟类', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车')

# --- 1. 加载 CIFAR-10 数据集 ---
dataset = ds.CIFAR10(root=r'../资源/', train=True, download=True)
print("训练样本：", len(dataset))     # 50000
dataset = ds.CIFAR10(root=r'../资源/', train=False, download=True)
print("测试样本：", len(dataset))     # 10000

# --- 2. 可视化样本 ---
ppl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
ppl.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
_, axes = ppl.subplots(5, 10)
for row in range(5):
    for col in range(10):
        n = row*10 + col
        img, label = dataset[n]
        print(label, img.width, img.height) # [32, 32]
        ax = axes[row][col]
        ax.imshow(img)
        ax.set_title(CLASS[label])  # ppl.title()
        ax.axis("off")     # 不显示坐标轴
ppl.show()