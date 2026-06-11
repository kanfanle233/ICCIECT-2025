"""
二元非线性分类训练示例。

教学重点：隐藏层让模型具备拟合复杂边界的能力。
"""

# 添加这行代码来解决 OMP 错误
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import matplotlib.pyplot as ppl

# 准备样本
x = np.random.uniform(-np.pi, np.pi, size=[500, 1])
print("x.shape=", x.shape)
y = np.sin(x)
print("y.shape=", y.shape)


class PolyModel(torch.nn.Module):
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank
        self.line = torch.nn.Linear(rank, out_features=1)

    def forward(self, x):
        poly = []
        power = 1
        for i in range(self.rank):
            power = (power * x) / (i + 1)
            poly.append(power)
        poly = torch.concat(poly, dim=1)
        y = self.line(poly)
        return y


model = PolyModel(30)
# 准备优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练
model.train()  # 模型进入训练模式
x = torch.tensor(x).float()
y = torch.tensor(y).float()
loss_fn = torch.nn.MSELoss()  # 均方差损失

for epoch in range(10000):  # 训练若干轮
    goal = model(x)
    loss = loss_fn(goal, y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # 这里的 'epoch' 变量在 for 循环中已经自动递增，不需要再加 1
    # epoch += 1

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch + 1:4d}: loss = {loss.item():.10f}')

# 把这行代码移到循环外面，确保只打印一次
print('训练完毕')

# ---
# 测试
x_test = np.linspace(-np.pi, np.pi, num=1000, dtype=np.float32)
y_test = np.sin(x_test)
model.eval()

with torch.no_grad():
    ps = 0
    for p in model.parameters():
        ps += p.numel()
        print(p.numpy())
    print("parameters:", ps)

    # 使用测试数据来计算预测结果
    p = torch.tensor(x_test.reshape([-1, 1]))
    p = model(p)
    p = p.numpy().reshape([-1])

# 绘图
ppl.plot(x_test, y_test, color="red", label="sin")
ppl.plot(x_test, p, color="blue", label="predict")
ppl.show()