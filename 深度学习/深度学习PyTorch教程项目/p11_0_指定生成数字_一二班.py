"""
MNIST 指定手写数字生成模型，使用条件变分自编码器（Conditional VAE / CVAE）。
教学重点：
1) CVAE 架构：编码器提取语义向量，解码器在每层注入类别标签信息
2) Merge 模块：将类别 one-hot 向量通过线性变换加到特征图上（条件注入）
3) 指定数字生成：给定语义向量和类别标签，生成对应数字图像
关于卷积神经网络，最好的电子教材之一：
https://cloud.tencent.com/developer/article/2109487
"""
from torchvision import datasets as ds, transforms as ts
from torch.utils.data import DataLoader
import torch
import numpy as np

import matplotlib.pyplot as ppl
import os
import cv2   # opencv-python

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
lr = 0.01
epochs = 2
model_path = "p11_0_指定生成数字_一二班.pth"
losses_size = 10

# --- 2. 准备样本 ---
# 1. 准备样本
tr = ts.Compose([
    ts.RandomAffine(0, (0.05, 0.05), (1, 1)),  # 随机平移增强
    ts.ToTensor()
])
dataset = ds.MNIST(root=r'../资源/', train=True, download=True,
                         transform=tr)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 3. 建模 ---
# 2. 建模
class MyConv(torch.nn.Module):
    """自定义卷积块：BatchNorm -> Conv2d -> ReLU -> MaxPool2d"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pool_size):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(pool_size, pool_size),
        )
    def forward(self, x):  # x: [?, c, h, w]
        return self.net(x)

class Merge(torch.nn.Module):
    """条件注入模块：将类别标签的 one-hot 向量通过线性变换加到特征图上"""
    def __init__(self, shape, size):
        super().__init__()
        total = np.prod(shape)  # 计算目标形状的总元素数
        self.net = torch.nn.Sequential(
            torch.nn.Linear(size, total),      # one-hot -> 特征图形状
            torch.nn.Unflatten(1, shape)       # 恢复空间形状
        )

    def forward(self, x, y_onehot):
        y = self.net(y_onehot)   # 类别信息映射为特征图
        return x + y             # 将类别信息叠加到特征图上

class MnistModel(torch.nn.Module):
    """条件变分自编码器（CVAE）：编码器提取语义，解码器在每层注入类别信息"""
    def __init__(self):
        super().__init__()
        # 编码器：[1, 28, 28] -> [4] 语义向量
        self.encode = torch.nn.Sequential(
            # [1, 28, 28] --> [8, 14, 14]
            MyConv(1, 8, 3, 1, 1, 2),
            # [8, 14, 14] --> [16, 7, 7]
            MyConv(8, 16, 3, 1, 1, 2),
            # [16, 7, 7] --> [4]
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 4,7, 7, 1),    # 大卷积核压缩空间维度

            torch.nn.BatchNorm2d(4, affine=False),
            torch.nn.Flatten()                   # 展平为 [4]
        )
        # 解码器：在每层注入类别信息
        self.merge0 = Merge([4], 10)             # 在语义向量层注入
        self.decode1 = torch.nn.Sequential(
            # [4] --> [16, 7, 7]
            torch.nn.Linear(4, 16*7*7),
            torch.nn.Unflatten(1, [16, 7, 7]),
            torch.nn.ReLU()
        )
        self.merge1 = Merge([16, 7, 7], 10)      # 在 16x7x7 层注入

        self.decode2 = torch.nn.Sequential(
            # [16, 7, 7] --> [8, 14, 14]
            torch.nn.BatchNorm2d(16),
            torch.nn.ConvTranspose2d(16, 8, 4, 2, 1),  # 反卷积
            torch.nn.ReLU()
        )
        self.merge2 = Merge([8, 14, 14], 10)      # 在 8x14x14 层注入

        self.decode3 = torch.nn.Sequential(
            # [8, 14, 14] --> [1, 28, 28]
            torch.nn.BatchNorm2d(8),
            torch.nn.ConvTranspose2d(8, 1, 4, 2, 1),   # 反卷积
        )
        self.merge3 = Merge([1, 28, 28], 10)      # 在输出层注入
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, y):  # [?, 1, 28, 28]
        v = self.encode(x)  # [?, 4] 语义向量
        p = self.decode(v, y)
        # [?, 1, 28, 28]
        return p

    def decode(self, v, y):
        """解码器：将语义向量和类别标签合成为图像"""
        y = torch.nn.functional.one_hot(y, 10)  # [?, 10] 转为独热编码
        y = y.float()
        p = self.merge0(v, y)    # 在语义向量层注入类别信息
        p = self.decode1(p)
        p = self.merge1(p, y)    # 在中间特征层注入
        p = self.decode2(p)
        p = self.merge2(p, y)
        p = self.decode3(p)
        p = self.merge3(p, y)    # 在输出层注入
        p = self.sigmoid(p)
        return p


def get_model():
    """加载或创建模型"""
    model = MnistModel().to(device)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        except:
            print(f'从{model_path}加载模型失败，很可能是模型结构发生了改变')
            exit(0)
    else:
        print(f'未发现模型{model_path}')
    return model

# --- 4. 训练函数 ---
def train():
    """训练 CVAE：输入图像和标签，目标是重建原图"""
    model = get_model()
    optimier = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # 3. 训练
    losses = []
    for epoch in range(epochs):
        for batch, (x, y) in enumerate(dataloader):
            model.train()
            x, y = x.to(device), y.to(device)
            # x: [?, 1, 28, 28], y: [?]
            p = model(x, y)
            # CVAE 不需要对 logits 调用 softmax，不需要把 y 转为独热向量
            loss = loss_fn(p, x)  # 重建图像 vs 原图

            loss.backward()
            optimier.step()
            optimier.zero_grad()

            model.eval()
            with torch.no_grad():
                losses.append(loss.item())
                if (batch + 1) % losses_size == 0:
                    loss = np.mean(losses)
                    print(f"epoch: {epoch+1}.{batch+1}, loss: {loss:.6f}")
                    losses.clear()
        torch.save(model.state_dict(), model_path)
        print(f'保存模型至{model_path}')
    print("训练完毕！")

# --- 5. 测试：生成指定数字的图像 ---
# 4. 测试
def _test():
    """生成 0-9 每个数字各两张，用 matplotlib 显示"""
    vectors = np.float32(np.random.normal(size=[20, 4]))  # 随机语义向量
    vectors = torch.from_numpy(vectors).to(device)

    y = np.int64(np.arange(10))
    y = np.reshape([y, y], [-1])  # 每个数字重复一次，共20个标签
    y = torch.from_numpy(y).to(device)

    model = get_model()
    model.eval()
    with torch.no_grad():
        ps = model.decode(vectors, y)    # [20, 1, 28, 28]
        ps = ps.cpu().numpy()
        ps = ps.reshape([20, 28, 28])
    for i, p in enumerate(ps):
        ppl.subplot(4, 5, i+1)
        p = np.uint8(p * 255)
        ppl.imshow(p)
        ppl.axis('off')
    ppl.show()

def _test_cv():
    """生成指定数字图像并用 OpenCV 拼接显示为网格"""
    vectors = np.float32(np.random.normal(size=[20, 4]))
    vectors = torch.from_numpy(vectors).to(device)

    y = np.int64(np.arange(10))
    y = np.reshape([y, y], [-1])  # 每个数字重复一次
    y = torch.from_numpy(y).to(device)

    model = get_model()
    model.eval()
    with torch.no_grad():
        ps = model.decode(vectors, y)
        ps = ps.cpu().numpy()     # [20, 1, 28, 28]
        ps = ps.reshape([20, 28, 28])

    # 拼接为 4行5列 网格
    ps = ps.reshape([4, 5, 28, 28])
    ps = ps.transpose([0, 2, 1, 3])     # --> [4, 28, 5, 28]
    ps = ps.reshape([4*28, 5*28])

    cv2.imshow('AAA', np.uint8(ps*255))
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # train()
    _test_cv()
