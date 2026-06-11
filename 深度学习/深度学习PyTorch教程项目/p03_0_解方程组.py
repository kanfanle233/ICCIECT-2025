"""
利用 PyTorch 自动微分求解二元一次方程组。
教学重点：将方程组转化为损失函数最小化问题，多参数同时优化。
"""

# 求解方程组：
# x + 3y = 4
# 4x - 2y = 1

import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = torch.nn.Parameter(torch.tensor(1.0))
        self.y = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self):
        return (self.x + 3 * self.y - 4,
                4 * self.x - 2 * self.y - 1)

# --- 1. 创建模型与优化器 ---
model = MyModel()
# 准备优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- 2. 训练：最小化方程组残差的平方和 ---
# 训练
model.train()               # 模型进入训练模式
for epoch in range(2000):      # 训练200轮
    goal1, goal2 = model()         # 构建预测张量（张量可以理解为公式）
    loss = goal1**2 + torch.square(goal2) # 构建损失张量
    loss.backward()         # 反向传播梯度（即根据损失优化模型的参数）
    optimizer.step()        # 优化器根据梯度张量优化模型的参数
    optimizer.zero_grad()   # 清空梯度。否则会影响下一次循环
    epoch += 1
    if epoch % 100 == 0:
        print(f'epoch {epoch:4d}: loss = {loss.item():.6f}')
print('训练完毕')

# 输出
x = model.x.detach().item()
y = model.y.detach().item()
print(f"x={x}")
print(f"y={y}")
print(x + 3 * y - 4)
print(4 * x - 2 * y - 1)

