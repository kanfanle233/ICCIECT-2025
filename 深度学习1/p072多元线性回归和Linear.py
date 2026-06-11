"""
用 torch.nn.Linear 实现多元线性回归。

教学重点：Linear 层本质上就是 y = xW + b。
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import matplotlib.pyplot as ppl  # 根据图片，使用 ppl 作为别名

# 1. 准备样本 (不变)
x_np = np.random.uniform(-2, 2, size=[500, 4])
# 真实权重 A = [[2], [-1], [3], [-2]]，真实偏置 b = [2]
y_np = np.matmul(x_np, [[2], [-1], [3], [-2]]) + [2]
y_np += np.random.normal(loc=0, scale=0.1, size=[500, 1])


# 2. 创建模型 (使用 nn.Linear)
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.line = torch.nn.Linear(in_features=4, out_features=1)

    def forward(self, x_forward):
        return self.line(x_forward)


model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 3. 训练 (按照图片，回归到批处理模式)
model.train()  # 模型进入训练模式

# 数据转换
x = torch.from_numpy(x_np).float()
y = torch.from_numpy(y_np).float()

losses = []
for epoch in range(2000):  # 训练2000轮
    goal = model(x)  # 一次性处理所有数据
    loss = torch.mean(torch.square(goal - y))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    losses.append(loss.item())

    # 注意: 图片中的 'epoch += 1' 是不必要的，因为 for 循环会自动递增 epoch，这里我们将其省略以保持代码简洁。

    if (epoch + 1) % 100 == 0:
        # 为了和图片中的epoch数值对应，这里使用 epoch+1
        print(f'epoch {(epoch + 1):4d}: loss = {loss.item():.10f}')

print('训练完毕')

# 4. 评估和打印参数 (完全按照图片新增的部分)
model.eval()  # 将模型切换到评估模式
print("\n--- 模型学到的参数 ---")
with torch.no_grad():  # 在此代码块中，禁用梯度计算，可以节省内存并加速
    for p in model.parameters():
        # .numpy() 需要在 CPU 上的 tensor 才能调用
        print(p.cpu().numpy())

    # 5. 可视化 (按照图片修改)
# np.arange(len(losses)) 生成 x 轴坐标 (0, 1, 2, ...)
ppl.plot(np.arange(len(losses)), losses)
ppl.title("Loss Curve")
ppl.xlabel("Epoch")
ppl.ylabel("Loss")
ppl.grid(True)
ppl.show()