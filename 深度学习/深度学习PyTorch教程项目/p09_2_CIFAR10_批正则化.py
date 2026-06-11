"""
CIFAR-10 十分类模型，加入批正则化（BatchNorm）与数据增强。
教学重点：
1) BatchNorm2d/BatchNorm1d 批正则化：加速训练收敛、提升稳定性
2) 数据增强 transforms：RandomCrop 随机裁剪、RandomHorizontalFlip 随机水平翻转
3) eval()/train() 与 Dropout、BatchNorm 的关系
4) 通道数扩展：3 -> 32 -> 64 -> 128
"""
import torch
import torch as T
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import time
from matplotlib import pyplot as ppl

batch_size = 100
lr = 0.001

# --- 1. 设备检测与超参数 ---
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
epochs = 2
model_path = "p09_2_model.pth"
moving_size = 10  # 滑动平均窗口大小

# --- 2. 准备样本（含数据增强）---
# 1) 样本
# 训练集使用数据增强：随机裁剪 + 随机水平翻转
transform = transforms.Compose([
    transforms.RandomCrop(32, 4),          # 先补边4像素，再随机裁剪回 32x32
    transforms.RandomHorizontalFlip(),     # 以 50% 概率水平翻转
    transforms.ToTensor()                  # 转为 [0,1] 范围的 Tensor
])

ds = datasets.CIFAR10(r'../资源', True, transform)
train_dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
ds = datasets.CIFAR10(r'../资源', False, transforms.ToTensor())  # 测试集不做增强
test_dl = DataLoader(ds, batch_size=10 * batch_size)

# --- 3. 建模（含 BatchNorm 的三层卷积网络）---
# 2) 建模
class Cifar10Model(T.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = T.nn.Sequential(
            T.nn.BatchNorm2d(3),             # 对输入的3个颜色通道分别进行批正则化

            # [3, 32, 32] --> [32, 16, 16]
            T.nn.Conv2d(3, 32, 3, 1, 1),     # 卷积：3通道 -> 32通道
            T.nn.BatchNorm2d(32),            # BatchNorm 通常位于卷积之后、激活之前
            T.nn.ReLU(),
            T.nn.MaxPool2d(2, 2),            # 池化：尺寸减半

            # [32, 16, 16] --> [64, 8, 8]
            T.nn.Conv2d(32, 64, 3, 1, 1),    # 卷积：32通道 -> 64通道
            T.nn.BatchNorm2d(64),
            T.nn.ReLU(),
            T.nn.MaxPool2d(2, 2),

            # [64, 8, 8] --> [128, 4, 4]
            T.nn.Conv2d(64, 128, 3, 1, 1),   # 卷积：64通道 -> 128通道
            T.nn.BatchNorm2d(128),
            T.nn.ReLU(),
            T.nn.MaxPool2d(2, 2),

            # [128, 4, 4] -- > [10]
            T.nn.Flatten(),                   # 展平为一维向量
            T.nn.Linear(128*4*4, 200),        # 全连接层
            T.nn.BatchNorm1d(200),            # 一维批正则化
            T.nn.ReLU(),
            T.nn.Dropout(0.5),               # Dropout 防止过拟合，训练时随机丢弃50%神经元
            # 不用调用T.nn.Softmax()，CrossEntropyLoss 内部已包含 Softmax
            T.nn.Linear(200, 10)    # 10个logits，对应10个类别
        )

    def forward(self, x):   # x: [?, 3, 32, 32]
        return self.net(x)  # [?, 10]

model = Cifar10Model().to(device)
optimizer = T.optim.Adam(model.parameters(), lr= lr)
print('参数数量：', sum([p.numel() for p in model.parameters()]))
# 尝试加载已有的模型权重
if model_path is not None and os.path.exists(model_path):
    try:
        model.load_state_dict(T.load(model_path, map_location=device))
    except Exception as e:
        print(f'从{model_path}加载模型失败, 很可能模型发生了改变')
        exit(0)
    print(f'从{model_path}加载模型成功')
else:
    print('没有发现老模型')

# --- 4. 测试函数 ---
# 3) 测试
def show_testing_acc():
    """在测试集上评估模型准确率"""
    model.eval()                # 切换到评估模式：固定 BatchNorm 统计量，关闭 Dropout
    with torch.no_grad():       # 推理时不需要计算梯度
        accs = []
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            accs.append(get_acc(logits, y))
    acc = np.mean(accs)
    print('测试精度: %.4f' % acc)

def get_acc(logits, y):
    """计算一个 batch 的准确率"""
    logits = T.argmax(logits, 1)  # 取概率最大的类别作为预测结果
    return (logits == y).float().mean().cpu().numpy()

# --- 5. 训练 ---
# 4) 训练
loss_fn = T.nn.CrossEntropyLoss().to(device)  # 交叉熵损失（含 Softmax）
total_loss, losses = [], []
total_acc, accs = [], []
for epoch in range(epochs):
    for batch, (x, y) in enumerate(train_dl):
        model.train()           # 切换到训练模式：启用 Dropout，更新 BatchNorm 统计量
        x, y = x.to(device), y.to(device)
        logits = model(x)       # 前向传播
        loss = loss_fn(logits, y)  # 计算损失

        loss.backward()         # 反向传播
        optimizer.step()        # 更新参数
        optimizer.zero_grad()   # 清空梯度

        # 切换到 eval 模式计算指标（避免 Dropout 和 BatchNorm 干扰）
        model.eval()
        with T.no_grad():
            losses.append(loss.item())
            accs.append(get_acc(logits, y))
            # 维护滑动窗口
            if len(losses) > moving_size:
                losses.pop(0)
                accs.pop(0)
            loss = np.mean(losses)  # 滑动平均损失
            acc = np.mean(accs)     # 滑动平均准确率
            print(f'{epoch+1}-{batch+1}: loss: {loss:.6f}, acc: {acc:.4f}')
            total_loss.append(loss)
            total_acc.append(acc)
    if model_path is not None:
        T.save(model.state_dict(), model_path)  # 每轮结束保存模型
        print('模型保存至', model_path)
    show_testing_acc()  # 每轮结束在测试集上验证
print('训练完毕')

# --- 6. 可视化损失和准确率曲线 ---
# 5）显示损失和准确度曲线
_, axes = ppl.subplots(1, 2)
axes[0].plot(total_loss)  # 左图：损失曲线
axes[1].plot(total_acc)   # 右图：准确率曲线
ppl.show()


