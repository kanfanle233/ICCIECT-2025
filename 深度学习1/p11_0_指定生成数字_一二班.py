"""
MNIST 指定手写数字生成模型，使用 Conditional VAE/条件自编码思想。

数据结构说明：
- images: [batch_size, 1, 28, 28]，输入手写数字图片。
- labels: [batch_size]，数字类别。
- one-hot labels: [batch_size, 10]，把类别编号改成 10 维向量，方便和图像特征融合。
- output: [batch_size, 1, 28, 28]，模型重建出来的图片。

关于卷积神经网络，最好的电子教材之一：
https://cloud.tencent.com/developer/article/2109487
"""
from torchvision import datasets as ds, transforms as ts
from torch.utils.data import DataLoader
import torch
import numpy as np

import matplotlib.pyplot as ppl
import os
from runtime_compat import (
    adapt_batch_size,
    build_loader_kwargs,
    get_best_device,
    mnist_root,
    move_to_device,
    print_device_summary,
)

batch_size = 8192
lr = 0.01
epochs = 20
model_path = "p11_0_指定生成数字_一二班.pth"
losses_size = 10
device = get_best_device()
batch_size = adapt_batch_size(batch_size, device, mps_cap=1024, cpu_cap=256)
print_device_summary(device)
print(f"batch_size: {batch_size}")

# 1. 准备样本
tr = ts.Compose([
    ts.RandomAffine(0, (0.05, 0.05), (1, 1)),
    ts.ToTensor()
])
dataset = ds.MNIST(root=mnist_root(), train=True, download=True,
                         transform=tr)
loader_kwargs = build_loader_kwargs(device, max_workers=4)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
print(f"DataLoader 参数: {loader_kwargs}")

# 2. 建模
class MyConv(torch.nn.Module):
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
    def __init__(self, shape, size):
        super().__init__()
        total = np.prod(shape)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(size, total),
            torch.nn.Unflatten(1, shape)
        )

    def forward(self, x, y_onehot):
        # 标签向量经过 Linear 变成和图像特征一样的形状，再相加。
        y = self.net(y_onehot)
        return x + y

class MnistModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
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

    def forward(self, images, labels):  # images: [?, 1, 28, 28]
        latent_vector = self.encode(images)  # [?, 4]
        label_onehot = torch.nn.functional.one_hot(labels, 10).float()  # [?, 10]
        p = self.merge0(latent_vector, label_onehot)
        p = self.decode1(p)
        p = self.merge1(p, label_onehot)
        p = self.decode2(p)
        p = self.merge2(p, label_onehot)
        p = self.decode3(p)
        p = self.merge3(p, label_onehot)
        p = self.sigmoid(p)
        # [?, 1, 28, 28]
        return p

def get_model():
    model = MnistModel().to(device)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as exc:
            print(f'从{model_path}加载模型失败，很可能是模型结构发生了改变：{exc}')
            raise SystemExit(1)
    else:
        print(f'未发现模型{model_path}')
    return model

def train():
    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # 3. 训练
    losses = []
    for epoch in range(epochs):
        for batch, (images, labels) in enumerate(dataloader):
            model.train()
            images, labels = move_to_device(images, labels, device=device)
            reconstructed_images = model(images, labels)
            # 自编码器的目标是让 reconstructed_images 尽量接近原始 images。
            loss = loss_fn(reconstructed_images, images)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

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

# 4. 测试
def _test():
    model = get_model()
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(dataloader))
        images, labels = move_to_device(images[:20], labels[:20], device=device)
        ps = model(images, labels).cpu().numpy()     # [20, 1, 28, 28]
        ps = ps.reshape([20, 28, 28])
    for i, p in enumerate(ps):
        ppl.subplot(4, 5, i+1)
        p = np.uint8(p * 255)
        ppl.imshow(p)
        ppl.axis('off')
    ppl.show()

if __name__ == '__main__':
    train()
