"""
用 PyTorch 求函数最小值，从而解方程组。

教学重点：loss 越小代表当前参数越接近目标解。
通过最小化两个方程残差的平方和，找到同时满足方程组的 x 和 y。
"""

# 求解方程组：
# x + 3y = 4
# 4x - 2y = 1

import torch


# --- 1. 定义模型 ---
class MyModel(torch.nn.Module):
    """将 x 和 y 设为可训练参数，forward 返回两个方程的残差。"""

    def __init__(self):
        super().__init__()
        # 初始猜测值都设为 1.0
        self.x = torch.nn.Parameter(torch.tensor(1.0))
        self.y = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self):
        # 返回两个方程的残差：越接近 0 表示方程越成立
        return (self.x + 3 * self.y - 4,
                4 * self.x - 2 * self.y - 1)


# --- 2. 训练循环 ---
model = MyModel()
# 准备优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练
model.train()                          # 模型进入训练模式
for epoch in range(2000):              # 训练 2000 轮
    goal1, goal2 = model()             # 前向传播，得到两个方程的残差
    loss = goal1**2 + torch.square(goal2)  # 损失 = 残差平方和，最小值为 0
    loss.backward()                    # 反向传播梯度（即根据损失优化模型的参数）
    optimizer.step()                   # 优化器根据梯度张量优化模型的参数
    optimizer.zero_grad()              # 清空梯度，否则会影响下一次循环
    epoch += 1
    if epoch % 100 == 0:
        print(f'epoch {epoch:4d}: loss = {loss.item():.6f}')
print('训练完毕')

# --- 3. 输出结果 ---
x = model.x.detach().item()
y = model.y.detach().item()
print(f"x={x}")
print(f"y={y}")
# 验证解是否满足方程：越接近 0 表示方程成立
print(x + 3 * y - 4)
print(4 * x - 2 * y - 1)
