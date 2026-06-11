"""
MNIST 风格迁移模型（一二班版），使用 CVAE 实现数字类别转换。
教学重点：
1) CVAE 条件注入：Merge 模块将类别 one-hot 信息注入解码器每层
2) 风格迁移推理：编码原图 -> 用目标类别解码 -> 生成新数字图像
3) OpenCV 拼接显示：将原图和迁移结果拼接为网格对比展示
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
model_path = "p12_0_指定生成数字_一二班.pth"
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
    """条件注入模块：将类别 one-hot 向量通过线性变换加到特征图"""
    def __init__(self, shape, size):
        super().__init__()
        total = np.prod(shape)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(size, total),
            torch.nn.Unflatten(1, shape)
        )

    def forward(self, x, y_onehot):
        y = self.net(y_onehot)
        return x + y

class MnistModel(torch.nn.Module):
    """CVAE 风格迁移模型：编码器提取语义向量，解码器注入目标类别"""
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
            torch.nn.Conv2d(16, 4,7, 7, 1),

            torch.nn.BatchNorm2d(4, affine=False),
            torch.nn.Flatten()
        )
        # 解码器：在每层注入类别信息
        self.merge0 = Merge([4], 10)
        self.decode1 = torch.nn.Sequential(
            # [4] --> [16, 7, 7]
            torch.nn.Linear(4, 16*7*7),
            torch.nn.Unflatten(1, [16, 7, 7]),
            torch.nn.ReLU()
        )
        self.merge1 = Merge([16, 7, 7], 10)

        self.decode2 = torch.nn.Sequential(
            # [16, 7, 7] --> [8, 14, 14]
            torch.nn.BatchNorm2d(16),
            torch.nn.ConvTranspose2d(16, 8, 4, 2, 1),
            torch.nn.ReLU()
        )
        self.merge2 = Merge([8, 14, 14], 10)

        self.decode3 = torch.nn.Sequential(
            # [8, 14, 14] --> [1, 28, 28]
            torch.nn.BatchNorm2d(8),
            torch.nn.ConvTranspose2d(8, 1, 4, 2, 1),
        )
        self.merge3 = Merge([1, 28, 28], 10)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, y, ty):  # [?, 1, 28, 28]
        """前向传播：x=原图, y=原类别, ty=目标类别"""
        v = self.encode(x)       # [?, 4] 提取语义向量
        p1 = self.decode(v, y)   # 原始类别重建

        p2 = self.decode(v, ty)  # 用目标类别解码（风格迁移）
        v = self.encode(p2)      # 再次编码
        p2 = self.decode(v, y)   # 用原始类别重建验证
        # [?, 1, 28, 28]
        return p1, p2

    def decode(self, v, y):
        """解码器：将语义向量和类别标签合成为图像"""
        y = torch.nn.functional.one_hot(y, 10)  # [?, 10] 转为独热编码
        y = y.float()
        p = self.merge0(v, y)
        p = self.decode1(p)
        p = self.merge1(p, y)
        p = self.decode2(p)
        p = self.merge2(p, y)
        p = self.decode3(p)
        p = self.merge3(p, y)
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
    """训练风格迁移模型"""
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
            ty = torch.randint(0, 10, y.shape, device=device)  # 随机目标类别
            p1, p2 = model(x, y, ty)
            # 两路损失：原始重建 + 风格迁移重建
            loss1 = loss_fn(p1, x)
            loss2 = loss_fn(p2, x)
            loss = loss1 + loss2

            loss.backward()
            optimier.step()
            optimier.zero_grad()

            model.eval()
            with torch.no_grad():
                losses.append([loss.item(), loss1.item(), loss2.item()])
                if (batch + 1) % losses_size == 0:
                    loss = np.mean(losses, 0)
                    print(f"epoch: {epoch+1}.{batch+1}, loss: {loss}")
                    losses.clear()
        torch.save(model.state_dict(), model_path)
        print(f'保存模型至{model_path}')
    print("训练完毕！")

# --- 5. 测试：风格迁移效果展示（OpenCV 拼接）---
# 4. 测试
def _test_cv():
    """从测试集取图，展示原图与风格迁移结果的对比（OpenCV 网格显示）"""
    dataset = ds.MNIST(root=r'../资源/', train=False, download=True,
                       transform=tr)
    x = [dataset[2597+i][0] for i in range(20)]  # 取20张测试图
    x = torch.from_numpy(np.float32(x)).to(device)

    ty = range(10)
    ty = np.array([ty, ty]).reshape([-1])  # ty:[20]，每个数字各两次
    ty = torch.from_numpy(ty).long().to(device)

    model = get_model()
    model.eval()
    with torch.no_grad():
        v = model.encode(x)          # 编码原图
        p = model.decode(v, ty)      # 用目标类别解码
        p = p.cpu().numpy()          # [20, 1, 28, 28]

    # 拼接：原图（上）+ 迁移结果（下）
    p = np.concatenate([x, p], 0)   # [40, 1, 28, 28]
    p = p.reshape([4, 10, 28, 28])  # 4行10列
    p = p.transpose([0, 2, 1, 3])  # --> [4, 28, 10, 28]
    p = p.reshape([4*28, 10*28])

    cv2.imshow('AAA', np.uint8(p*255))
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # train()
    _test_cv()
