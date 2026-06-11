"""
CIFAR-10 彩色图片分类进阶版：加入数据增强、BatchNorm 和滑动平均曲线。

数据结构说明：
- train_transform 会随机裁剪和翻转训练图片，让模型见到更多变化。
- total_loss_for_plot / total_acc_for_plot 保存滑动平均后的指标，曲线会更平滑。
- epoch_results 是字典列表，最后保存成 CSV，方便课后分析每一轮效果。

设备选择：
自动按 MPS(Mac) -> CUDA -> CPU 运行。
"""

import torch
import torch as T
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd  # <--- 1. 导入 pandas 用于保存CSV
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
batch_size = 1000
lr = 0.001
epochs = 2  # <--- 建议增加轮数以观察模型效果
model_path = "p091_model.pth"
result_path = "p093_result.csv"  # <--- 结果保存路径

# --- 2. 数据增强和转换 ---
# 训练集使用数据增强
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 标准化
])

# 测试集不使用数据增强，但需要标准化
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# --- 3. 设备选择 ---
device = get_best_device()
batch_size = adapt_batch_size(batch_size, device, mps_cap=512, cpu_cap=256)
loader_kwargs = build_loader_kwargs(device, max_workers=4)
print_device_summary(device)
print(f"当前 batch_size: {batch_size}")
print(f"DataLoader 参数: {loader_kwargs}")

# --- 4. 数据加载 (修正了参数和 transform) ---
train_dataset = datasets.CIFAR10(
    root=cifar10_root(), train=True, download=True, transform=train_transform  # <--- 修正1：应用训练变换
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
test_dataset = datasets.CIFAR10(
    root=cifar10_root(), train=False, download=True, transform=test_transform  # <--- 修正2：应用测试变换
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)


# --- 5. 建模 (定义模型结构) ---
class Cifar10Model(T.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = T.nn.Sequential(
            # BatchNorm 让每一层输入更稳定，初学者可以把它理解成“自动校准特征范围”。
            T.nn.BatchNorm2d(3),
            T.nn.Conv2d(3, 32, 3, 1, 1), T.nn.BatchNorm2d(32), T.nn.ReLU(), T.nn.MaxPool2d(2, 2),
            T.nn.Conv2d(32, 64, 3, 1, 1), T.nn.BatchNorm2d(64), T.nn.ReLU(), T.nn.MaxPool2d(2, 2),
            T.nn.Conv2d(64, 128, 3, 1, 1), T.nn.BatchNorm2d(128), T.nn.ReLU(), T.nn.Dropout(0.5), T.nn.MaxPool2d(2, 2),
            T.nn.Flatten(),
            T.nn.Linear(128 * 4 * 4, 400),
            T.nn.BatchNorm1d(400),
            T.nn.ReLU(),
            T.nn.Linear(400, 10),
        )

    def forward(self, x):
        return self.net(x)


# --- 6. 加载或初始化模型 ---
model = Cifar10Model().to(device)
optimizer = T.optim.Adam(model.parameters(), lr=lr)
loss_fn = T.nn.CrossEntropyLoss()
print('参数数量:', sum([p.numel() for p in model.parameters()]))


# (加载模型的逻辑不变)

# --- 7. 辅助函数 (精度计算和测试) ---
def get_acc(logits, y):
    """计算一个 batch 的准确率。"""
    predicted = T.argmax(logits, dim=1)
    return (predicted == y).float().mean().item()


def get_testing_acc():
    # ... (此函数内容不变)
    model.eval()
    with torch.no_grad():
        accs = []
        for images, labels in test_loader:
            images, labels = move_to_device(images, labels, device=device)
            logits = model(images)
            accs.append(get_acc(logits, labels))
    model.train()
    return np.mean(accs)


# --- 8. 训练循环 (集成滑动平均逻辑) ---
total_loss_for_plot = []
total_acc_for_plot = []
epoch_results = []  # 用于保存每轮的最终结果

# --- 新增：为滑动平均做准备 ---
moving_size = 50  # 定义滑动窗口大小
losses = []  # 存储最近的loss
accs = []  # 存储最近的acc

print("开始训练...")
for epoch in range(epochs):
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = move_to_device(images, labels, device=device)

        # 训练核心步骤
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        # ==========================================================
        # 核心修改：集成图片中的滑动平均逻辑
        # ==========================================================
        # 1. 将当前批次的瞬时 loss 和 acc 存入列表
        losses.append(loss.item())
        accs.append(get_acc(logits, labels))

        # 2. 如果列表超过了窗口大小，就移除最旧的那个
        if len(losses) > moving_size:
            losses.pop(0)
            accs.pop(0)

        # 3. 计算滑动平均值
        moving_loss = np.mean(losses)
        moving_acc = np.mean(accs)

        # 4. 将滑动平均值存入用于绘图的总列表
        total_loss_for_plot.append(moving_loss)
        total_acc_for_plot.append(moving_acc)

        # 5. 打印滑动平均值
        if (batch_idx + 1) % 10 == 0 or batch_idx == len(train_loader) - 1:
            print(
                f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_loader)} | Moving Loss: {moving_loss:.4f} | Moving Acc: {moving_acc:.4f}")
        # ==========================================================
        # 修改结束
        # ==========================================================

    test_acc_epoch = get_testing_acc()
    epoch_results.append({'epoch': epoch + 1, 'final_loss': moving_loss, 'test_accuracy': test_acc_epoch})
    print(f"--- Epoch {epoch + 1} 结束 --- Test Accuracy: {test_acc_epoch:.4f}\n")

print('训练完毕')

# --- 9. 保存模型和结果 ---
torch.save(model.state_dict(), model_path)
print(f"模型已保存到: {model_path}")

# --- 新增：保存结果到CSV ---
df = pd.DataFrame(epoch_results)
df.to_csv(result_path, index=False)
print(f"每轮测试结果已保存到: {result_path}")

# --- 10. 显示损失和准确度曲线 ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(total_loss_for_plot)
plt.title('Moving Average Loss per Batch')
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.subplot(1, 2, 2)
plt.plot(total_acc_for_plot, color='orange')
plt.title('Moving Average Accuracy per Batch')
plt.xlabel('Batch Number')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()
