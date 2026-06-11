"""
MNIST 自编码器示例。

自编码器的目标不是分类，而是“先压缩图片，再还原图片”。
数据结构：
- 输入 images: [batch_size, 1, 28, 28]
- encode 输出 latent: [batch_size, 4, 1, 1]，可以理解为压缩后的 4 个核心特征。
- decode 输出 reconstruction: [batch_size, 1, 28, 28]
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as ppl
from runtime_compat import (
    build_loader_kwargs,
    get_best_device,
    mnist_root,
    move_to_device,
    print_device_summary,
)

# —— 设置中文字体（防止字体警告）——
ppl.rcParams['font.sans-serif'] = ['SimHei']
ppl.rcParams['axes.unicode_minus'] = False

# ========================
# 1. 基本参数配置
# ========================
batch_size = 100
lr = 0.01
epochs = 2
model_path = 'p100_.pth'
losses_size = 10
device = get_best_device()
loader_kwargs = build_loader_kwargs(device, max_workers=4)
print_device_summary(device)
print(f"DataLoader 参数: {loader_kwargs}")

# ========================
# 2. 数据集加载
# ========================
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root=mnist_root(), train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)

# ========================
# 3. 模型定义
# ========================
class MnistAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(
            # 编码器：图片越来越小、通道越来越多，表示“压缩并提取关键特征”。
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 4, 7),
            nn.ReLU(),
        )
        self.decode = nn.Sequential(
            # 解码器：用反卷积把 4 个核心特征逐步还原成 28x28 图片。
            nn.ConvTranspose2d(4, 16, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out

# ========================
# 4. 实例化模型
# ========================
model = MnistAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f'✅ 已成功从 {model_path} 加载模型参数')
    except Exception as e:
        print(f'⚠️ 模型加载失败：{e}')
else:
    print('未发现已有模型，开始从头训练')

# ========================
# 5. 训练过程
# ========================
losses = []
for epoch in range(epochs):
    model.train()
    for batch_idx, (images, _) in enumerate(train_loader, start=1):
        images = move_to_device(images, device=device)
        reconstruction = model(images)
        loss = loss_fn(reconstruction, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if batch_idx % losses_size == 0:
            avg_loss = np.mean(losses[-losses_size:])
            print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}] | Loss: {avg_loss:.6f}")

print("✅ 训练完毕")

# torch.save(model.state_dict(), model_path)

# ========================
# 6. 随机潜向量生成（整合截图内容）
# ========================
model.eval()
# 随机生成 20 个潜向量
vectors = np.float32(np.random.normal(size=[20, 4]))
vectors = move_to_device(torch.from_numpy(vectors).unsqueeze(-1).unsqueeze(-1), device=device)  # [20, 4, 1, 1]

with torch.no_grad():
    ps = model.decode(vectors).cpu().numpy()  # [20, 1, 28, 28]
    ps = ps.reshape([20, 28, 28])

# —— 绘制伪彩色图像（类似你截图的效果）—— #
ppl.figure(figsize=(8, 6))
for i, p in enumerate(ps):
    ppl.subplot(4, 5, i + 1)
    ppl.imshow(p, cmap='viridis')  # 使用彩色渐变显示
    ppl.axis('off')

ppl.suptitle("随机潜向量生成结果", fontsize=14)
ppl.tight_layout()
ppl.show()
