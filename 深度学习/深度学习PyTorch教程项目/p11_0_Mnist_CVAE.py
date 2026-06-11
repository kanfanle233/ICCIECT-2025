"""
MNIST 指定数字生成模型，使用条件变分自编码器（Conditional VAE / CVAE）。
教学重点：
1) CVAE 架构：编码器提取语义向量，解码器在每层注入类别标签信息
2) Merge 模块：将类别 one-hot 向量通过线性变换加到特征图上（条件注入）
3) 指定数字生成：给定语义向量和类别标签，生成对应数字图像
"""
from torchvision import datasets as ds, transforms as ts
from torch.utils.data import DataLoader
import torch as T
import numpy as np
import os
import matplotlib.pyplot as ppl

batch_size = 100
lr = 0.01
epochs = 20
model_path = 'p11_0_model.pth'
losses_size = 10

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

# --- 1. 准备样本 ---
# 1. 准备样本
tr = ts.Compose([
    ts.RandomAffine(10, (1/7, 1/7), (0.9, 1.1)),
    ts.ToTensor()
])

dataset = ds.MNIST(root=r'../资源/', train=True, download=True,
                         transform=tr)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 2. 建模（CVAE 条件变分自编码器）---
# 2. 建模
class MyConv(T.nn.Module):
    """自定义卷积块：BatchNorm -> Conv2d -> ReLU -> MaxPool2d"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pool_size):
        super().__init__()
        self.net = T.nn.Sequential(
            T.nn.BatchNorm2d(in_channels),
            T.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            T.nn.ReLU(),
            T.nn.MaxPool2d(pool_size, pool_size),
        )

    def forward(self, x):
        return self.net(x)

class Merge(T.nn.Module):
    """条件注入模块：将类别 one-hot 向量通过线性变换加到特征图上"""
    def __init__(self, shape, y_size):
        super().__init__()
        t_size = np.prod(shape)  # 计算目标形状的总元素数
        self.net = T.nn.Sequential(
            T.nn.Linear(y_size, t_size),   # one-hot -> 特征图大小
            T.nn.Unflatten(1, shape)       # 恢复空间形状
        )
    def forward(self, p, y_onehot):
        y = self.net(y_onehot)   # 类别信息映射为特征图
        return y + p             # 将类别信息叠加到特征图上


class MnistModel(T.nn.Module):
    """CVAE 条件变分自编码器：编码器提取语义，解码器在每层注入类别信息"""
    def __init__(self):
        super().__init__()
        self.encode = T.nn.Sequential(
            # [1, 28, 28] --> [32, 28, 28]
            MyConv(1, 32, 3, 1, 1, 1),
            # [32, 28, 28] --> [64, 14, 14]
            MyConv(32, 64, 3, 1, 1, 2),
            # [64, 14, 14] --> [128, 7, 7]
            MyConv(64, 128, 3, 1, 1, 2),
            # [64, 7, 7] --> [4, 1, 1]
            T.nn.BatchNorm2d(128),
            T.nn.Conv2d(128, 4, 7, 1),

            T.nn.BatchNorm2d(4, affine=False),
            T.nn.Flatten()
        )

        self.merge0 = Merge([4], 10)
        self.decode1 = T.nn.Sequential(
            # [4, 1, 1] --> [64, 7, 7]
            T.nn.Linear(4, 64*7*7),
            T.nn.Unflatten(1, [64, 7, 7]),
            T.nn.ReLU()
        )
        self.merge1 = Merge([64, 7, 7], 10)
        self.decode2 = T.nn.Sequential(
            # [64, 7, 7] --> [32, 14, 14]
            T.nn.BatchNorm2d(64),
            T.nn.ConvTranspose2d(64, 32, 4, 2, 1),
            T.nn.ReLU()
        )
        self.merge2 = Merge([32, 14, 14], 10)
        self.decode3 = T.nn.Sequential(
            # [32, 14, 14] --> [1, 28, 28]
            T.nn.BatchNorm2d(32),
            T.nn.ConvTranspose2d(32, 1, 4, 2, 1)
        )
        self.merge3 = Merge([1, 28, 28], 10)
        self.sigmoid = T.nn.Sigmoid()

    def forward(self, x, y):
        # x: [?, 1, 28, 28]
        # y: [?, 10]
        p = self.encode(x)  # [?, 4] 语义向量
        p = self.decode(p, y)
        return p

    def decode(self, p, y):
        y = T.nn.functional.one_hot(y, 10)
        y = y.float()
        p = self.merge0(p, y)
        p = self.decode1(p)
        p = self.merge1(p, y)
        p = self.decode2(p)
        p = self.merge2(p, y)
        p = self.decode3(p)
        p = self.merge3(p, y)
        p = self.sigmoid(p)
        return p


# --- 3. 模型加载 ---
def get_model():
    model = MnistModel()
    model = model.to(device)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(T.load(model_path, weights_only=True, map_location=device))
        except:
            print(f'从{model_path}加载模型失败, 很可能是因为模型发生了改变')
            exit(0)
    else:
        print(f'未发现模型{model_path}')
    return model

# --- 4. 训练函数 ---
def train():
    """训练 CVAE：输入图像和标签，目标是重建原图"""
    model = get_model()
    optimier = T.optim.Adam(model.parameters(), lr=lr)
    loss_fn = T.nn.MSELoss().to(device)

    # 3. 训练
    losses = []
    for epoch in range(epochs):
        for batch, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            model.train()
            # x: [?, 1, 28, 28], y: [?, 1, 28, 28]
            # print('x min, max=', np.max(x.numpy()))
            p = model(x, y)
            loss = loss_fn(p, x)

            loss.backward()
            optimier.step()
            optimier.zero_grad()

            model.eval()
            with T.no_grad():
                losses.append(loss.cpu().item())
                loss = np.mean(losses)
                if (batch+1) % losses_size == 0:
                    print(f"epoch: {epoch+1}.{batch+1}, loss: {loss:.6f}")
                    losses.clear()
        T.save(model.state_dict(), model_path)
        print(f'保存模型{model_path}')
    print("训练完毕！")

# --- 5. 测试：生成指定数字的图像 ---
# 4. 测试
def _test():
    """生成 0-9 每个数字各一张"""
    model = get_model()
    num = 10
    vec = np.random.normal(size=(10, 4))
    vec = T.from_numpy(np.float32(vec)).to(device)
    y = np.arange(10, dtype=np.int32)
    y = T.from_numpy(y).long().to(device)
    model.eval()
    with T.no_grad():
        imgs = model.decode(vec, y)    # [10, 1, 28, 28]
        imgs = imgs.cpu().numpy()
        imgs = np.reshape(imgs, [10, 28, 28])
        imgs = np.uint8(imgs*255)
    for i, img in enumerate(imgs):
        ppl.subplot(2, 5, i+1)
        ppl.imshow(img)
        ppl.axis('off')
    ppl.show()

if __name__ == '__main__':
    # train()
    _test()