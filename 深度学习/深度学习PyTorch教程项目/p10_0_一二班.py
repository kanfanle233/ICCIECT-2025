"""
MNIST 手写数字生成模型（一二班版），使用卷积自编码器。
教学重点：
1) 编码器 Encoder：Conv2d + BatchNorm2d + ReLU + MaxPool2d 压缩至4维语义向量
2) 解码器 Decoder：Linear + Unflatten + ConvTranspose2d 反卷积重建图像
3) BCELoss 二元交叉熵重建损失
4) 随机采样生成：从标准正态分布采样，经解码器生成数字图像
"""
from torchvision import datasets as ds, transforms as ts
from torch.utils.data import DataLoader
import torch as T
import numpy as np
import os
import matplotlib.pyplot as ppl

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

batch_size = 100
lr = 0.01
epochs = 20
model_path = 'p10_0_model.pth'
losses_size = 10  # 每隔多少个 batch 打印一次

# --- 2. 准备样本（含数据增强）---
# 1. 准备样本
tr = ts.Compose([
    ts.RandomAffine(10, (1/7, 1/7), (0.9, 1.1)),  # 随机旋转10度、平移1/7、亮度变化
    ts.ToTensor()
])

dataset = ds.MNIST(root=r'../资源/', train=True, download=True,
                         transform=tr)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 3. 建模（卷积自编码器）---
# 2. 建模
class MnistModel(T.nn.Module):
    """卷积自编码器：编码器压缩图像至4维语义向量，解码器重建原始图像"""
    def __init__(self):
        super().__init__()
        # 编码器：[1, 28, 28] -> [4] 语义向量
        self.encode = T.nn.Sequential(
            # [1, 28, 28] --> [32, 28, 28]
            T.nn.BatchNorm2d(1),
            T.nn.Conv2d(1, 32, 3, 1, 1),
            T.nn.ReLU(),
            # T.nn.MaxPool2d(2, 2),

            # [32, 28, 28] --> [64, 14, 14]
            T.nn.BatchNorm2d(32),
            T.nn.Conv2d(32, 64, 3, 1, 1),
            T.nn.ReLU(),
            T.nn.MaxPool2d(2, 2),          # 尺寸减半

            # [64, 14, 14] --> [128, 7, 7]
            T.nn.BatchNorm2d(64),
            T.nn.Conv2d(64, 128, 3, 1, 1),
            T.nn.ReLU(),
            T.nn.MaxPool2d(2, 2),

            # [128, 7, 7] --> [4, 1, 1]（大卷积核空间压缩）
            T.nn.BatchNorm2d(128),
            T.nn.Conv2d(128, 4, 7, 1),

            T.nn.BatchNorm2d(4, affine=False),  # 无可学习参数的 BatchNorm
            T.nn.Flatten()                       # 展平为 [4]
        )

        # 解码器：[4] -> [1, 28, 28] 重建图像
        self.decode = T.nn.Sequential(
            # [4] --> [16, 7, 7]
            T.nn.Linear(4, 128*7*7),             # 全连接展开
            T.nn.Unflatten(1, [128, 7, 7]),      # 恢复空间形状
            T.nn.ReLU(),

            # [128, 7, 7] --> [64, 14, 14]
            T.nn.BatchNorm2d(128),
            T.nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 反卷积：尺寸翻倍
            T.nn.ReLU(),

            # [64, 14, 14] --> [32, 28, 28]
            T.nn.BatchNorm2d(64),
            T.nn.ConvTranspose2d(64, 32, 4, 2, 1),
            T.nn.ReLU(),

            # [32, 28, 28] --> [1, 28, 28]
            T.nn.BatchNorm2d(32),
            T.nn.Conv2d(32, 1, 3, 1, 1),
            T.nn.Sigmoid()  # 输出 [0,1] 像素值
        )

    def forward(self, x):  # [?, 1, 28, 28]
        y = self.encode(x)  # [?, 4] 语义向量
        y = self.decode(y)  # [?, 1, 28, 28] 重建图像
        return y

model = MnistModel().to(device)
# 尝试加载已有模型
if os.path.exists(model_path):
    try:
        model.load_state_dict(T.load(model_path, weights_only=True, map_location=device))
    except:
        print(f'从{model_path}加载模型失败, 很可能是因为模型发生了改变')
        exit(0)
else:
    print(f'未发现模型{model_path}')

# --- 4. 训练函数 ---
def train():
    """训练自编码器：输入原图，目标也是原图，最小化重建误差"""
    optimier = T.optim.Adam(model.parameters(), lr=lr)
    loss_fn = T.nn.BCELoss()  # 二元交叉熵损失（适用于 Sigmoid 输出）

    # 3. 训练
    losses = []
    for epoch in range(epochs):
        for batch, (x, _) in enumerate(dataloader):  # 标签不需要
            model.train()
            x = x.to(device)
            # x: [?, 1, 28, 28], y: [?, 1, 28, 28]
            # print('x min, max=', np.max(x.numpy()))
            y = model(x)
            loss = loss_fn(y, x)  # 重建图像 vs 原图

            loss.backward()
            optimier.step()
            optimier.zero_grad()

            model.eval()
            with T.no_grad():
                losses.append(loss.item())
                loss = np.mean(losses)
                if (batch+1) % losses_size == 0:
                    print(f"epoch: {epoch+1}.{batch+1}, loss: {loss:.6f}")
                    losses.clear()
        T.save(model.state_dict(), model_path)
        print(f'保存模型{model_path}')
    print("训练完毕！")

# --- 5. 测试：随机采样生成数字图像 ---
# 4. 测试
def _test():
    """从标准正态分布采样，用解码器生成新数字图像"""
    vec = np.random.normal(size=(20, 4))            # 采样20个4维向量
    vec = T.from_numpy(np.float32(vec)).to(device)
    model.eval()
    with T.no_grad():
        imgs = model.decode(vec)    # [20, 1, 28, 28]，仅用解码器
        imgs = imgs.cpu().numpy()
        imgs = np.reshape(imgs, [20, 28, 28])
        imgs = np.uint8(imgs*255)   # 转为 0-255 像素值
    for i, img in enumerate(imgs):
        ppl.subplot(4, 5, i+1)
        ppl.imshow(img)
        ppl.axis('off')
    ppl.show()

if __name__ == '__main__':
    # _test()
    train()
