"""
MNIST 手写数字生成模型，使用卷积自编码器（Conv AutoEncoder）。
教学重点：
1) 编码器 Encoder：Conv2d 卷积 + MaxPool2d 池化，逐步压缩图像至低维语义向量
2) 解码器 Decoder：ConvTranspose2d 反卷积（转置卷积），逐步恢复图像尺寸
3) BCELoss/MSELoss 重建损失：衡量生成图像与原图的差异
4) 随机采样生成：从正态分布采样向量，经解码器生成新图像
"""
from torchvision import datasets as ds, transforms as ts
from torch.utils.data import DataLoader
import torch
import numpy as np

import matplotlib.pyplot as ppl
import os

# --- 1. 设备检测与超参数 ---
# ── 设备检测（优先 MPS，其次 CUDA，最后 CPU）──────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("使用 MPS (Apple Silicon GPU) 加速")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("使用 CUDA 加速")
else:
    device = torch.device("cpu")
    print("使用 CPU 训练")

batch_size = 100
lr = 0.001
epochs = 20
model_path = "p10_0_.pth"
losses_size = 10  # 每隔多少个 batch 打印一次损失

# --- 2. 准备样本（含随机平移增强）---
# 1. 准备样本
tr = ts.Compose([
    ts.RandomAffine(0, (0.05, 0.05)),   # 随机平移增强（旋转角度0，平移比例5%）
    ts.ToTensor()                        # 转为 [0,1] 范围的 Tensor
])
dataset = ds.MNIST(root=r'../资源/', train=True, download=True,
                         transform=tr)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 3. 建模（卷积自编码器）---
# 2. 建模
class MnistModel(torch.nn.Module):
    """卷积自编码器：编码器压缩图像至8维语义向量，解码器重建原始图像"""
    def __init__(self):
        super().__init__()
        # 编码器：逐步压缩图像 [1, 28, 28] -> [8]
        self.encode = torch.nn.Sequential(
            # [1, 28, 28] --> [8, 14, 14]
            torch.nn.BatchNorm2d(1),
            torch.nn.Conv2d(1, 8, 3, 1, 1),   # 卷积：1通道 -> 8通道
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),          # 池化：28x28 -> 14x14

            # [8, 14, 14] --> [16, 7, 7]
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, 16, 3, 1, 1),   # 卷积：8通道 -> 16通道
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),          # 池化：14x14 -> 7x7

            # [16, 7, 7] --> [8, 1, 1]（使用 7x7 卷积核将空间维度压缩为 1x1）
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 8,7, 7, 1),    # 大卷积核卷积：空间压缩

            torch.nn.BatchNorm2d(8, affine=False),  # 无可学习参数的 BatchNorm
            torch.nn.Flatten()                       # 展平为 [8] 维向量
        )

        # 解码器：从语义向量恢复图像 [8] -> [1, 28, 28]
        self.decode = torch.nn.Sequential(
            # [8] --> [16, 7, 7]
            torch.nn.Linear(8, 16*7*7),              # 全连接展开
            torch.nn.Unflatten(1, [16, 7, 7]),       # 恢复空间形状
            torch.nn.ReLU(),
            # [16, 7, 7] --> [8, 14, 14]
            torch.nn.BatchNorm2d(16),
            torch.nn.ConvTranspose2d(16, 8, 4, 2, 1),  # 反卷积：尺寸翻倍
            torch.nn.ReLU(),
            # [8, 14, 14] --> [1, 28, 28]
            torch.nn.BatchNorm2d(8),
            torch.nn.ConvTranspose2d(8, 1, 4, 2, 1),   # 反卷积：尺寸翻倍
            torch.nn.Sigmoid()                           # 输出 [0,1] 像素值
        )

    def forward(self, x):  # [?, 1, 28, 28]
        p = self.encode(x)  # [?, 8] 语义向量
        p = self.decode(p)  # [?, 1, 28, 28] 重建图像
        return p

model = MnistModel().to(device)
# 尝试加载已有模型
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    except:
        print(f'从{model_path}加载模型失败，很可能是模型结构发生了改变')
        exit(0)
else:
    print(f'未发现模型{model_path}')

# --- 4. 训练函数 ---
def train():
    """训练自编码器：以原图作为输入和目标，最小化重建误差"""
    optimier = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()  # 均方误差损失

    # 3. 训练
    losses = []
    for epoch in range(epochs):
        for batch, (x, _) in enumerate(dataloader):  # 标签 _ 不需要使用
            model.train()
            x = x.to(device)
            # x: [?, 1, 28, 28], y: [?]
            p = model(x)
            # 自编码器：输入 = 目标，不需要 softmax，不需要独热编码
            loss = loss_fn(p, x)  # 重建图像与原图的均方误差

            loss.backward()
            optimier.step()
            optimier.zero_grad()

            model.eval()
            with torch.no_grad():
                losses.append(loss.item())
                if (batch + 1) % losses_size == 0:
                    loss = np.mean(losses)
                    print(f"epoch: {epoch+1}.{batch+1}, loss: {loss:.6f}")
        torch.save(model.state_dict(), model_path)
        print(f'保存模型至{model_path}')
    print("训练完毕！")

# --- 5. 测试：随机采样生成数字图像 ---
# 4. 测试
# train()

# 从标准正态分布采样20个8维向量
vectors = np.float32(np.random.normal(size=[20, 8]))
vectors = torch.from_numpy(vectors).to(device)
model.eval()
with torch.no_grad():
    ps = model.decode(vectors)      # 仅用解码器生成图像
    ps = ps.cpu().numpy()           # [20, 1, 28, 28]
    ps = ps.reshape([20, 28, 28])
# 用 matplotlib 显示生成结果
for i, p in enumerate(ps):
    ppl.subplot(4, 5, i+1)
    p = np.uint8(p * 255)           # 转为 0-255 像素值
    ppl.imshow(p)
    ppl.axis('off')
ppl.show()

# 用 OpenCV 拼接显示（可选）
import cv2  # opencv-python
ps = np.uint8(ps * 255) # [20, 28, 28]
ps = np.reshape(ps, [4, 5, 28, 28])
ps = np.transpose(ps, [0, 2, 1, 3]) # [4, 28, 5, 28]，重组为网格布局
ps = np.reshape(ps, [4*28, 5*28])
cv2.imshow('AAA', ps)
cv2.waitKey()
cv2.destroyAllWindows()
