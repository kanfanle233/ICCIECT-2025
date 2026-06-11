"""
MNIST 手写数字分类入门示例。

数据结构说明：
- images: [batch_size, 1, 28, 28]，一批灰度数字图片。
- labels: [batch_size]，每张图片对应的真实数字 0-9。
- logits: [batch_size, 10]，模型对 10 个数字类别给出的分数。

运行设备：
通过 runtime_compat 自动选择 MPS -> CUDA -> CPU，学生不用手动改代码。
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from runtime_compat import (
    build_loader_kwargs,
    get_best_device,
    mnist_root,
    move_to_device,
    print_device_summary,
)

# 1. 准备数据
batch_size = 100
lr = 0.01
epochs = 2
device = get_best_device()
loader_kwargs = build_loader_kwargs(device, max_workers=4)
print_device_summary(device)
print(f"DataLoader 参数: {loader_kwargs}")

# 训练集
train_dataset = datasets.MNIST(
    root=mnist_root(),
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)

# 测试集
test_dataset = datasets.MNIST(
    root=mnist_root(),
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

# 2. 建立模型
class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 卷积层会在小窗口里学习笔画特征，比如横线、竖线、弯钩。
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Flatten 把特征图拉成一行，最后用 Linear 输出 10 个数字类别分数。
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 10)  # 输入图像28x28，池化后是14x14
        )

    def forward(self, x):
        return self.net(x)

# 3. 训练
model = MnistModel().to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

all_losses = []

for epoch in range(epochs):
    epoch_losses = []  # 保存当前 epoch 的所有 loss
    for batch_index, (images, labels) in enumerate(train_loader):
        images, labels = move_to_device(images, labels, device=device)
        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录 loss
        batch_loss = loss.item()
        all_losses.append(batch_loss)
        epoch_losses.append(batch_loss)

    # 每个 epoch 打印一次平均 loss
    print(f"Epoch {epoch+1}, Avg Loss: {np.mean(epoch_losses):.4f}")

# 4. 测试
model.eval()
acc = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = move_to_device(images, labels, device=device)
        logits = model(images)
        predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
        true_labels = labels.detach().cpu().numpy()
        batch_accuracy = np.mean(predictions == true_labels)
        acc.append(batch_accuracy)

print(f'准确率: {np.mean(acc):.4f}')

# 5. 可视化训练损失
plt.plot(all_losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

