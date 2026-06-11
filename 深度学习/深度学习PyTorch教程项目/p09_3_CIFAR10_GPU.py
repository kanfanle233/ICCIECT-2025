"""
CIFAR-10 十分类模型（GPU 大 batch 训练版）。
教学重点：
1) 样本变形：RandomCrop 补边裁剪 + RandomHorizontalFlip 随机翻转
2) GPU 训练：model.to(device)、x.to(device)、loss.cpu().item()
3) 大 batch 训练策略：batch_size=1000，配合数据增强
"""
import torch
import torch as T
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import time
# from matplotlib import pyplot as ppl

batch_size = 1000
lr = 0.001
epochs = 200
model_path = "p09_3_gpu_model.pth"
moving_size = 10
print_size = 10
resource_path = '.'

# --- 1. 准备样本（含数据增强）---
# 1) 样本
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 补边4，之后随机剪裁成32*32
    transforms.RandomHorizontalFlip(),          # 随机水平翻转
    transforms.ToTensor()                       # 转为张量
])
ds = datasets.CIFAR10(resource_path, True, transform)
train_dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
ds = datasets.CIFAR10(resource_path, False, transforms.ToTensor())
test_dl = DataLoader(ds, batch_size=10 * batch_size)

# --- 2. 建模（含 BatchNorm 的卷积网络）---
# 2) 建模
class Cifar10Model(T.nn.Module):
    """三层卷积网络 + BatchNorm：适合 GPU 大 batch 训练"""
    def __init__(self):
        super().__init__()
        self.net = T.nn.Sequential(
            T.nn.BatchNorm2d(3),        # 在3个颜色上分别进行批正则化

            # [3, 32, 32] --> [32, 16, 16]，彩色图的第一个卷积通常输出32个通道
            T.nn.Conv2d(3, 32, 3, 1, 1),
            T.nn.BatchNorm2d(32),       # BatchNorm操作通常位于卷积之后，激活之前
            T.nn.ReLU(),
            T.nn.MaxPool2d(2, 2),

            # [32, 16, 16] --> [64, 8, 8]
            T.nn.Conv2d(32, 64, 3, 1, 1),
            T.nn.BatchNorm2d(64),
            T.nn.ReLU(),
            T.nn.MaxPool2d(2, 2),

            # [64, 8, 8] --> [128, 4, 4]
            T.nn.Conv2d(64, 128, 3, 1, 1),
            T.nn.BatchNorm2d(128),
            T.nn.ReLU(),
            T.nn.Dropout(0.7),                      #
            T.nn.MaxPool2d(2, 2),

            # [128, 4, 4] --> [512] --> [10]
            T.nn.Flatten(),
            T.nn.Linear(128*4*4, 400),    # 增加了一层Linear
            T.nn.BatchNorm1d(400),
            T.nn.ReLU(),
            # 不用调用T.nn.Softmax()
            T.nn.Linear(400, 10)    # 10个logits
        )

    def forward(self, x):   # x: [?, 3, 32, 32]
        return self.net(x)  # [?, 10]

model = Cifar10Model()
optimizer = T.optim.Adam(model.parameters(), lr= lr)
print('参数数量：', sum([p.numel() for p in model.parameters()]))
if model_path is not None and os.path.exists(model_path):
    try:
        model.load_state_dict(T.load(model_path, weights_only=True, map_location=device))
    except Exception as e:
        print(f'从{model_path}加载模型失败, 很可能模型发生了改变')
        exit(0)
    print(f'从{model_path}加载模型成功')
else:
    print('没有发现老模型')

# --- 3. 测试函数 ---
# 3) 测试
def show_testing_acc(device):
    model.eval()
    with torch.no_grad():
        accs = []
        for x, y in test_dl:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            accs.append(get_acc(logits, y))
    acc = np.mean(accs)
    print('测试精度: %.4f' % acc)

def get_acc(logits, y):
    logits = T.argmax(logits, 1)
    return np.mean((logits == y).cpu().numpy())

# --- 4. 设备检测与训练 ---
# 4) 训练
loss_fn = T.nn.CrossEntropyLoss()
total_loss, losses = [], []
total_acc, accs = [], []

# ── 设备检测（优先 MPS，其次 CUDA，最后 CPU）──────────────
if T.backends.mps.is_available():
    device = T.device("mps")
    print("使用 MPS (Apple Silicon GPU) 加速")
elif T.cuda.is_available():
    device = T.device("cuda")
    print("使用 CUDA 加速")
else:
    device = T.device("cpu")
    print("使用 CPU 训练")
model = model.to(device)

for epoch in range(epochs):
    for batch, (x, y) in enumerate(train_dl):
        x = x.to(device)
        y = y.to(device)
        model.train()           # 移动
        logits = model(x)
        loss = loss_fn(logits, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        model.eval()          # 新增
        with T.no_grad():
            losses.append(loss.cpu().item())
            accs.append(get_acc(logits, y))

            if len(losses) > moving_size:
                losses.pop(0)
                accs.pop(0)
            loss = np.mean(losses)
            acc = np.mean(accs)
            if (batch+1) % print_size == 0:
                print(f'{epoch+1}-{batch+1}: loss: {loss:.6f}, acc: {acc:.6f}')
            total_loss.append(loss)
            total_acc.append(acc)

    show_testing_acc(device)      # 移动
    if model_path is not None:
        T.save(model.state_dict(), model_path)
        print('模型保存至', model_path)

print('训练完毕')
np.save('total_loss', np.array(total_loss))
np.save('total_acc', np.array(total_acc))
# --- 5. 可视化 ---
# 5）显示损失和准确度曲线
# _, axes = ppl.subplots(1, 2)
# axes[0].plot(total_loss)
# axes[1].plot(total_acc)
# ppl.show()




