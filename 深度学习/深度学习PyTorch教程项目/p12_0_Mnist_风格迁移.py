"""
MNIST 风格迁移模型，通过编码器-解码器实现数字风格变换。
教学重点：
1) CVAE 条件解码器：在解码器每层注入目标类别信息（Merge 模块）
2) 风格迁移推理：编码原图得到语义向量，用目标类别解码生成新风格图像
3) 两路损失：原始类别重建损失 + 目标类别重建损失（风格迁移训练）
"""
from torchvision import datasets as ds, transforms as ts
from torch.utils.data import DataLoader
import torch as T
import numpy as np
import os
import matplotlib.pyplot as ppl

batch_size = 5
lr = 0.01
epochs = 20
model_path = 'p12_0_model.pth'
losses_size = 10

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

# --- 2. 准备样本 ---
# 1. 准备样本
tr = ts.Compose([
    ts.RandomAffine(0, (1/28, 1/28), (1.0, 1.0)),  # 随机平移（约1个像素）
    ts.ToTensor()
])

dataset = ds.MNIST(root=r'../资源/', train=True, download=True,
                         transform=tr)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 3. 建模 ---
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
    """条件注入模块：将类别 one-hot 向量映射并叠加到特征图"""
    def __init__(self, shape, y_size):
        super().__init__()
        t_size = np.prod(shape)
        self.net = T.nn.Sequential(
            T.nn.Linear(y_size, t_size),
            T.nn.Unflatten(1, shape)
        )
    def forward(self, p, y_onehot):
        y = self.net(y_onehot)
        return y + p


class MnistModel(T.nn.Module):
    """CVAE 风格迁移模型：编码器提取语义，解码器支持多类别条件注入"""
    def __init__(self):
        super().__init__()
        # 编码器：[1, 28, 28] -> [4] 语义向量
        self.encode = T.nn.Sequential(
            # [1, 28, 28] --> [32, 14, 14]
            MyConv(1, 32, 3, 1, 1, 2),
            # [32, 14, 14] --> [64, 7, 7]
            MyConv(32, 64, 3, 1, 1, 2),
            # [64, 7, 7] --> [4, 1, 1]
            # [64, 7, 7] --> [4, 1, 1]
            T.nn.BatchNorm2d(64),
            T.nn.Conv2d(64, 4, 7, 1),

            T.nn.BatchNorm2d(4, affine=False),
            T.nn.Flatten()                   # 展平为 [4]
        )

        # 解码器：在每层注入类别信息
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
            T.nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 反卷积
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

    def forward(self, x, y, ty):
        """风格迁移前向传播：
        x: 原始图像, y: 原始类别, ty: 目标类别
        返回：p1(原始类别重建), p2(目标类别风格迁移结果)
        """
        # x: [?, 1, 28, 28]
        # y: [?, 10]
        v = self.encode(x)       # [?, 4] 提取语义向量
        p2 = self.decode(v, ty)  # 用目标类别解码 -> 风格迁移图像
        p2 = self.encode(p2)     # 再次编码，提取迁移后语义
        p2 = self.decode(p2, y)  # 用原始类别解码 -> 重建验证

        p1 = self.decode(v, y)   # 用原始类别解码 -> 原始重建

        return p1, p2

    def decode(self, p, y):
        """解码器：将语义向量和类别标签合成为图像"""
        y = T.nn.functional.one_hot(y, 10)  # 转为独热编码
        y = y.float()
        p = self.merge0(p, y)    # 在语义层注入
        p = self.decode1(p)
        p = self.merge1(p, y)    # 在 64x7x7 层注入
        p = self.decode2(p)
        p = self.merge2(p, y)    # 在 32x14x14 层注入
        p = self.decode3(p)
        p = self.merge3(p, y)    # 在输出层注入
        p = self.sigmoid(p)
        return p


def get_model():
    """加载或创建模型"""
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
    """训练风格迁移模型：同时优化原始重建和目标风格迁移"""
    model = get_model()
    optimier = T.optim.Adam(model.parameters(), lr=lr)
    loss_fn = T.nn.MSELoss().to(device)

    # 3. 训练
    losses = []
    for epoch in range(epochs):
        for batch, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            ty = get_target_y(y.shape[0]).to(device)  # 随机生成目标类别
            model.train()
            # x: [?, 1, 28, 28], y: [?, 1, 28, 28]
            p1, p2 = model(x, y, ty)
            loss1 = loss_fn(p1, x)   # 原始类别重建损失
            loss2 = loss_fn(p2, x)   # 风格迁移重建损失
            loss = loss1 + loss2     # 总损失

            loss.backward()
            optimier.step()
            optimier.zero_grad()

            model.eval()
            with T.no_grad():
                losses.append([loss.cpu().item(), loss1.cpu().item(), loss2.cpu().item()])
                loss = np.mean(losses, 0)
                if (batch+1) % losses_size == 0:
                    print(f"epoch: {epoch+1}.{batch+1}, loss: {loss}")
                    losses.clear()
        T.save(model.state_dict(), model_path)
        print(f'保存模型{model_path}')
    print("训练完毕！")

def get_target_y(num):
    """随机生成目标类别标签"""
    return T.randint(0, 10, [num])

# --- 5. 测试：风格迁移效果展示 ---
# 4. 测试
def _test():
    """生成指定数字图像（0-9）"""
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
    train()
    # _test()
