"""
多元线性回归和 DataLoader 示例。

教学重点：多列特征如何组成输入矩阵，以及 batch 训练为什么更稳定。
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader  # 导入Dataloader相关工具

# 1. 准备样本 (不变)
x = np.random.uniform(-2, 2, size=[500, 4])
y = np.matmul(x, [[2], [-1], [3], [-2]]) + [2]
y += np.random.normal(loc=0, scale=0.1, size=[500, 1])


# 2. 创建模型 (不变)
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p0 = torch.nn.Parameter(torch.tensor(0.1))
        self.p1 = torch.nn.Parameter(torch.tensor(0.1))
        self.p2 = torch.nn.Parameter(torch.tensor(0.1))
        self.p3 = torch.nn.Parameter(torch.tensor(0.1))
        self.pb = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, x_forward):
        pred = (self.p0 * x_forward[:, 0] +
                self.p1 * x_forward[:, 1] +
                self.p2 * x_forward[:, 2] +
                self.p3 * x_forward[:, 3] +
                self.pb)
        return pred.unsqueeze(1)


model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 3. 数据处理优化：使用 DataLoader (新增核心优化)
# 将 numpy 数据转为 tensor
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

# 将特征和标签打包成一个数据集
dataset = TensorDataset(x_tensor, y_tensor)

# 定义 DataLoader
# batch_size=32: 每次从数据集中取出32个样本进行训练
# shuffle=True: 在每个 epoch 开始时，打乱数据顺序，这有助于提高模型泛化能力
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. 训练循环优化 (修改为按批次训练)
model.train()
losses = []  # 用于记录每个epoch的平均loss
num_epochs = 50  # 由于参数更新更频繁，通常不需要那么多epoch

print("--- 开始小批量训练 ---")
for epoch in range(num_epochs):
    epoch_loss = 0.0  # 记录当前 epoch 的总 loss
    # 循环从 DataLoader 中取出每个小批量
    for batch_x, batch_y in data_loader:
        # batch_x 的形状是 [32, 4], batch_y 的形状是 [32, 1]

        # 前向传播
        goal = model(batch_x)
        # 计算损失
        loss = torch.mean(torch.square(goal - batch_y))

        # 反向传播和优化
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()  # 累加这个批次的loss

    # 计算并记录当前 epoch 的平均 loss
    avg_loss = epoch_loss / len(data_loader)
    losses.append(avg_loss)

    # 打印训练信息
    if (epoch + 1) % 5 == 0:
        print(f'Epoch {(epoch + 1):2d}/{num_epochs}: Average Loss = {avg_loss:.10f}')

print('训练完毕')

# 5. 验证与可视化 (不变)
print("\n--- 结果验证 ---")
print(f"真实权重 (A): [2, -1, 3, -2]")
learned_weights = [model.p0.item(), model.p1.item(), model.p2.item(), model.p3.item()]
print(f"模型学到的权重: {[f'{w:.4f}' for w in learned_weights]}")
print(f"\n真实偏置 (b): [2]")
print(f"模型学到的偏置: {model.pb.item():.4f}")

plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title("Loss Decrease Curve (Mini-batch Training)")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)
plt.show()