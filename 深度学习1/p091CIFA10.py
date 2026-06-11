"""
CIFAR-10 彩色图片分类基础版。

数据结构说明：
- images: [batch_size, 3, 32, 32]，3 表示 RGB 三个颜色通道。
- labels: [batch_size]，每张图片的类别编号，范围是 0-9。
- logits: [batch_size, 10]，模型对 10 个类别输出的分数。

设备选择：
runtime_compat 会按 MPS(Mac) -> CUDA -> CPU 的顺序自动选择。
"""

import torch
import torch as T
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
from matplotlib import pyplot as plt
from runtime_compat import (
    adapt_batch_size,
    build_loader_kwargs,
    cifar10_root,
    get_best_device,
    move_to_device,
    print_device_summary,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 1. 超参数设置 ---
batch_size = 6000
lr = 0.001
epochs = 10
model_path = "p091_model.pth"

# --- 2. 设备选择 ---
device = get_best_device()
batch_size = adapt_batch_size(batch_size, device, mps_cap=512, cpu_cap=256)
loader_kwargs = build_loader_kwargs(device, max_workers=4)
print_device_summary(device)
print(f"当前 batch_size: {batch_size}")
print(f"DataLoader 参数: {loader_kwargs}")

# --- 3. 数据加载 ---
train_dataset = datasets.CIFAR10(
    root=cifar10_root(), train=True, download=True, transform=transforms.ToTensor()
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
test_dataset = datasets.CIFAR10(
    root=cifar10_root(), train=False, download=True, transform=transforms.ToTensor()
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)


# --- 4. 建模 (定义模型结构，添加 Dropout) ---
class Cifar10Model(T.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = T.nn.Sequential(
            # 卷积层部分：每一组 Conv + ReLU + Pool 都会提取更抽象的图像特征。
            T.nn.Conv2d(3, 16, 3, 1, 1), T.nn.ReLU(), T.nn.MaxPool2d(2, 2),
            T.nn.Conv2d(16, 32, 3, 1, 1), T.nn.ReLU(), T.nn.MaxPool2d(2, 2),
            T.nn.Conv2d(32, 64, 3, 1, 1), T.nn.ReLU(), T.nn.MaxPool2d(2, 2),
            T.nn.Flatten(),

            # 全连接层部分：把卷积特征映射成 10 个类别分数。
            T.nn.Linear(64 * 4 * 4, 200),
            T.nn.ReLU(),
            T.nn.Dropout(0.5),  # Dropout 随机关闭一部分神经元，帮助减少过拟合。
            T.nn.Linear(200, 10),
        )

    def forward(self, x):
        return self.net(x)


# --- 5. 加载或初始化模型 ---
model = Cifar10Model().to(device)
optimizer = T.optim.Adam(model.parameters(), lr=lr)
loss_fn = T.nn.CrossEntropyLoss()
print('参数数量:', sum([p.numel() for p in model.parameters()]))

if model_path is not None and os.path.exists(model_path):
    try:
        model.load_state_dict(T.load(model_path, map_location=device))
        print(f"成功从 {model_path} 加载模型参数")
    except Exception as e:
        print(f"从 {model_path} 加载模型失败, 错误信息: {e}")
        print("将从头开始训练。")
else:
    print("没有发现已保存的模型，将从头开始训练。")


# --- 6. 辅助函数 (精度计算和测试) ---
def get_acc(logits, y):
    """计算一个 batch 的分类准确率。"""
    predicted = T.argmax(logits, dim=1)
    return (predicted == y).float().mean().item()


def get_testing_acc():
    model.eval()  # Dropout层在eval模式下会自动关闭
    with torch.no_grad():
        accs = []
        for images, labels in test_loader:
            images, labels = move_to_device(images, labels, device=device)
            logits = model(images)
            accs.append(get_acc(logits, labels))
    model.train()  # 测试结束后切回训练模式
    return np.mean(accs)


# --- 7. 训练循环 ---
total_batch_loss = []
total_batch_acc = []
epoch_test_accs = []

print("开始训练...")
for epoch in range(epochs):
    model.train()  # Dropout层在train模式下会自动开启
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = move_to_device(images, labels, device=device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_batch_loss.append(loss.item())
        batch_acc = get_acc(logits, labels)
        total_batch_acc.append(batch_acc)

        if (batch_idx + 1) % 10 == 0 or batch_idx == len(train_loader) - 1:
            print(
                f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f} | Train Acc: {batch_acc:.4f}")

    test_acc_epoch = get_testing_acc()
    epoch_test_accs.append(test_acc_epoch)
    print(f"--- Epoch {epoch + 1} 结束 --- Test Accuracy: {test_acc_epoch:.4f}\n")

print('训练完毕')

# --- 8. 保存模型 ---
torch.save(model.state_dict(), model_path)
print(f"模型已保存到: {model_path}")

# --- 9. 显示损失和准确度曲线 ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(total_batch_loss)
plt.title('Training Loss per Batch')
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.subplot(1, 2, 2)
plt.plot(total_batch_acc, color='orange')
plt.title('Training Accuracy per Batch')
plt.xlabel('Batch Number')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()
