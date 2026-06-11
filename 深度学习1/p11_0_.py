"""
Conditional AutoEncoder for MNIST 指定数字生成。

教学重点：
- images: [batch_size, 1, 28, 28]，MNIST 灰度图。
- labels: [batch_size]，告诉模型"这张图是什么数字"。
- latent z: [batch_size, 4]，把图片压缩成 4 个核心数字特征。
- one-hot 标签: [batch_size, 10]，把数字类别变成模型容易使用的向量。

设备会自动选择 MPS -> CUDA -> CPU；训练结束后统一显示重建和生成效果。
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from torchvision import datasets as ds, transforms as ts
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from runtime_compat import (
    build_loader_kwargs,
    get_best_device,
    mnist_root,
    move_to_device,
    print_device_summary,
)


# ======================== 超参数 ========================
batch_size = 100
lr = 1e-3
epochs = 2
latent_dim = 4
model_path = "p11_0_指定生成数字_一二班.pth"
device = get_best_device()
print_device_summary(device)

# ======================== 数据 ========================
transform = ts.Compose([
    ts.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    ts.ToTensor()
])
dataset = ds.MNIST(root=mnist_root(), train=True, download=True, transform=transform)
loader_kwargs = build_loader_kwargs(device, max_workers=2)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
print(f"DataLoader 参数: {loader_kwargs}")

# ======================== 模型结构 ========================
class MyConv(torch.nn.Module):
    """基础卷积模块：BatchNorm -> Conv2d -> ReLU -> MaxPool。"""
    def __init__(self, in_c, out_c, k, s, p, pool):
        super().__init__()
        self.net = torch.nn.Sequential(
            # BatchNorm 让每个通道的数值范围更稳定，Conv2d 负责提取笔画特征。
            torch.nn.BatchNorm2d(in_c),
            torch.nn.Conv2d(in_c, out_c, k, s, p),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(pool, pool)
        )
    def forward(self, x): return self.net(x)

class Merge(torch.nn.Module):
    """将标签 one-hot 向量映射为和特征同形状的张量，再与特征相加。"""
    def __init__(self, shape, num_cls=10):
        super().__init__()
        total = int(np.prod(shape))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(num_cls, total),
            torch.nn.Unflatten(1, shape)
        )
    def forward(self, feat, onehot):
        # 把标签信息变成和特征同形状的张量，再相加，相当于告诉模型"要生成哪个数字"。
        return feat + self.net(onehot)

class MnistCVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = torch.nn.Sequential(
            MyConv(1, 8, 3, 1, 1, 2),
            MyConv(8, 16, 3, 1, 1, 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, latent_dim, 7, 1, 0),
            torch.nn.BatchNorm2d(latent_dim, affine=False),
            torch.nn.Flatten()
        )
        self.merge0 = Merge([latent_dim])
        self.decode1 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 16*7*7),
            torch.nn.Unflatten(1, (16, 7, 7)),
            torch.nn.ReLU()
        )
        self.merge1 = Merge([16, 7, 7])
        self.decode2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(16),
            torch.nn.ConvTranspose2d(16, 8, 4, 2, 1),
            torch.nn.ReLU()
        )
        self.merge2 = Merge([8, 14, 14])
        self.decode3 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(8),
            torch.nn.ConvTranspose2d(8, 1, 4, 2, 1)
        )
        self.merge3 = Merge([1, 28, 28])
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, y):
        z = self.encode(x)
        y_onehot = torch.nn.functional.one_hot(y, 10).float()
        return self._decode(z, y_onehot)

    def _decode(self, z, y_onehot):
        p = self.merge0(z, y_onehot)
        p = self.decode1(p)
        p = self.merge1(p, y_onehot)
        p = self.decode2(p)
        p = self.merge2(p, y_onehot)
        p = self.decode3(p)
        p = self.merge3(p, y_onehot)
        return self.sigmoid(p)

    def generate(self, z, y):
        y_onehot = torch.nn.functional.one_hot(y, 10).float()
        return self._decode(z, y_onehot)

# ---------------- 工具函数 ----------------
def get_model():
    model = MnistCVAE().to(device)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded weights from {model_path}")
        except Exception as e:
            print("载入失败:", e)
    else:
        print("No pretrained model, training from scratch.")
    return model

# ---------- 可视化（仅显示不保存） ----------
@torch.no_grad()
def show_reconstruction(model):
    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = move_to_device(images[:10], labels[:10], device=device)
    reconstruction = model(images, labels).cpu()
    imgs = torch.cat([images.cpu(), reconstruction], dim=0).squeeze(1)
    plt.figure(figsize=(10, 3))
    for i, img in enumerate(imgs):
        plt.subplot(2, 10, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        if i == 0: plt.title("Original")
        if i == 10: plt.title("Reconstruction")
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def show_generation(model):
    model.eval()
    z = torch.randn(10, latent_dim, device=device)
    labels = torch.arange(10, device=device)
    gen = model.generate(z, labels).cpu().squeeze(1)
    plt.figure(figsize=(10, 1.5))
    for i, img in enumerate(gen):
        plt.subplot(1, 10, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(str(i))
    plt.tight_layout()
    plt.show()

# ---------- 训练 ----------
def train():
    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    loss_history = []

    for ep in range(1, epochs + 1):
        model.train()
        for step, (images, labels) in enumerate(dataloader):
            images, labels = move_to_device(images, labels, device=device)
            reconstruction = model(images, labels)
            loss = loss_fn(reconstruction, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            if (step + 1) % 100 == 0:
                print(f"Epoch {ep}/{epochs} | Step {step + 1} | Loss {loss.item():.5f}")

        torch.save(model.state_dict(), model_path)
        print(f"Epoch {ep} done, model saved.")

    # ---- 全部训练结束后一次性显示 ----
    plt.figure(figsize=(6, 3))
    plt.plot(loss_history)
    plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Training Loss Curve")
    plt.tight_layout()
    plt.show()

    show_reconstruction(model)
    show_generation(model)

# ---------- 主入口 ----------
if __name__ == "__main__":
    train()
