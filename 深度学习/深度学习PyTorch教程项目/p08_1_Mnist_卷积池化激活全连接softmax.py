"""
MNIST 手写数字识别模型，使用三层卷积网络实现十分类。
教学重点：Conv2d 卷积层、BatchNorm2d 批正则化、MaxPool2d 池化层、
         Flatten 展平、Linear 全连接层、Dropout 防过拟合、
         CrossEntropyLoss 损失函数、DataLoader 批量加载、训练与测试流程。
利用 Apple Silicon MPS 加速 + 更大batch提升性能
"""
import os
import time
from torchvision import datasets as ds, transforms as ts
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as ppl

# ── MPS 设备检测 ──────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("使用 MPS (Metal Performance Shaders) 加速")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("使用 CUDA 加速")
else:
    device = torch.device("cpu")
    print("使用 CPU 训练")

# --- 1. 超参数设置 ---
# ── 超参数（16GB 内存可开大batch） ──────────────────────────
batch_size = 512       # M5 16GB 可以跑 512，显存不足可降到 256
lr = 0.001             # 大batch时 lr 适当降低
epochs = 10            # 多跑几轮，充分训练

# ── 数据路径检测 ──────────────────────────────────────────
if os.path.exists('../资源/MNIST/raw/'):
    root = '../资源/'
elif os.path.exists('../MNIST/raw/'):
    root = '../'
else:
    root = '../资源/'

# --- 2. 准备样本（含归一化）---
# 1. 准备样本
transform = ts.Compose([
    ts.ToTensor(),
    ts.Normalize((0.1307,), (0.3081,))  # MNIST 全局均值/标准差归一化
])
dataset = ds.MNIST(root=root, train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 3. 建模（三层卷积网络）---
# 2. 建模（加深加宽，利用 GPU 算力）
class MnistModel(torch.nn.Module):
    """三层卷积网络：Conv2d + BatchNorm + ReLU + MaxPool2d x 3 -> Flatten -> Linear -> 10类"""
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            # [1, 28, 28] --> [32, 14, 14]
            torch.nn.Conv2d(1, 32, 3, 1, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            # [32, 14, 14] --> [64, 7, 7]
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            # [64, 7, 7] --> [128, 3, 3]
            torch.nn.Conv2d(64, 128, 3, 1, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            # [128, 3, 3] --> [10]
            torch.nn.Flatten(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128*3*3, 10)
        )

    def forward(self, x):
        return self.net(x)

model = MnistModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()

# --- 4. 训练 ---
# 3. 训练
model.train()
total_losses = []
t0 = time.time()
for epoch in range(epochs):
    losses = []
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        avg_loss = np.mean(losses)
        total_losses.append(avg_loss)

        if (batch + 1) % 20 == 0:
            print(f"epoch {epoch+1}/{epochs}  batch {batch+1:3d}  loss: {avg_loss:.6f}")

    # 每轮结束打印一次
    print(f"── epoch {epoch+1} 完成, 平均 loss: {np.mean(losses):.6f}")

print(f"训练完毕！耗时 {time.time()-t0:.1f}s")

# --- 5. 测试评估 ---
# 4. 测试
@torch.no_grad()  # 装饰器：整个函数内禁用梯度计算
def show_accuracy(dl, title):
    model.eval()
    correct = total = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        p = model(x).argmax(1)
        correct += (p == y).sum().item()
        total += y.size(0)
    print(f'{title}准确率: {correct/total:.6f}')

show_accuracy(dataloader, "训练")

test_ds = ds.MNIST(root=root, train=False, download=True, transform=transform)
test_dl = DataLoader(test_ds, batch_size=batch_size)
show_accuracy(test_dl, "测试")

ppl.scatter(np.arange(len(total_losses)), total_losses)
ppl.show()






