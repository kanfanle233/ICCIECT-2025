"""
CIFAR-10 十分类模型（GPU 版），展示在 GPU 上使用批正则化和 Dropout。
教学重点：
1) eval() 和 no_grad() 的作用：关闭 Dropout、固定 BatchNorm 统计量
2) model.to(device) 将模型移至 GPU
3) loss.cpu().item() 将 GPU 上的标量取回 CPU
"""
import torch
import torch as T
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import time
# from matplotlib import pyplot as ppl

batch_size = 100
lr = 0.001
epochs = 2000
model_path = "p09_3_gpu_model.pth"
moving_size = 10  # 滑动平均窗口大小
print_size = 40   # 每隔多少个 batch 打印一次
resource_path = '.'

# --- 1. 准备样本 ---
# 1) 样本
ds = datasets.CIFAR10(resource_path, True, transforms.ToTensor())
train_dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
ds = datasets.CIFAR10(resource_path, False, transforms.ToTensor())
test_dl = DataLoader(ds, batch_size=10 * batch_size)

# --- 2. 建模（含 BatchNorm 的卷积网络）---
# 2) 建模
class Cifar10Model(T.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = T.nn.Sequential(
            # [3, 32, 32] --> [16, 16, 16]
            T.nn.Conv2d(3, 16, 3, 1, 1),     # 卷积：3通道 -> 16通道
            T.nn.BatchNorm2d(16),            # 批正则化：稳定训练过程
            T.nn.ReLU(),
            T.nn.MaxPool2d(2, 2),            # 最大池化：尺寸减半

            # [16, 16, 16] --> [32, 8, 8]
            T.nn.Conv2d(16, 32, 3, 1, 1),
            T.nn.BatchNorm2d(32),
            T.nn.ReLU(),
            T.nn.MaxPool2d(2, 2),

            # [32, 8, 8] --> [64, 4, 4]
            T.nn.Conv2d(32, 64, 3, 1, 1),
            T.nn.BatchNorm2d(64),
            T.nn.ReLU(),
            T.nn.Dropout(0.5),               # Dropout 防止过拟合
            T.nn.MaxPool2d(2, 2),

            # [64, 4, 4] -- > [10]
            T.nn.Flatten(),
            T.nn.Linear(64*4*4, 512),        # 增加了一层 Linear，扩大隐藏层
            T.nn.BatchNorm1d(512),
            T.nn.ReLU(),
            # 不用调用T.nn.Softmax()
            T.nn.Linear(512, 10)    # 10个logits
        )

    def forward(self, x):   # x: [?, 3, 32, 32]
        return self.net(x)  # [?, 10]

model = Cifar10Model()
optimizer = T.optim.Adam(model.parameters(), lr= lr)
print('参数数量：', sum([p.numel() for p in model.parameters()]))
# 尝试加载已有的模型权重
if model_path is not None and os.path.exists(model_path):
    try:
        model.load_state_dict(T.load(model_path, weights_only=True))
    except Exception as e:
        print(f'从{model_path}加载模型失败, 很可能模型发生了改变')
        exit(0)
    print(f'从{model_path}加载模型成功')
else:
    print('没有发现老模型')

# --- 3. 测试函数 ---
# 3) 测试
def show_testing_acc(device):
    """在测试集上评估模型准确率"""
    model.eval()                # 评估模式：固定 BatchNorm、关闭 Dropout
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
    """计算一个 batch 的准确率"""
    logits = T.argmax(logits, 1)  # 取概率最大的类别
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
model = model.to(device)  # 将模型移至 GPU

for epoch in range(epochs):
    for batch, (x, y) in enumerate(train_dl):
        x = x.to(device)    # 数据移至 GPU
        y = y.to(device)
        model.train()           # 训练模式
        logits = model(x)
        loss = loss_fn(logits, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        model.eval()            # 切换到评估模式计算指标
        with T.no_grad():
            losses.append(loss.cpu().item())  # GPU 标量取回 CPU
            accs.append(get_acc(logits, y))

            if len(losses) > moving_size:
                losses.pop(0)
                accs.pop(0)
            loss = np.mean(losses)  # 滑动平均损失
            acc = np.mean(accs)     # 滑动平均准确率
            if (batch+1) % print_size == 0:
                print(f'{epoch+1}-{batch+1}: loss: {loss:.6f}, acc: {acc:.4f}')
            total_loss.append(loss)
            total_acc.append(acc)

    show_testing_acc(device)      # 每轮结束在测试集上验证
    if model_path is not None:
        T.save(model.state_dict(), model_path)
        print('模型保存至', model_path)

print('训练完毕')

# --- 5. 可视化 ---
# 5）显示损失和准确度曲线
_, axes = ppl.subplots(1, 2)
axes[0].plot(total_loss)
axes[1].plot(total_acc)
ppl.show()


