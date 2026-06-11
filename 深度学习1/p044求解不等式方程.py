"""
用 PyTorch 联合处理方程和不等式，求带约束条件的最大值。

教学重点：多个目标可以合成一个总 loss。
不等式约束通过引入松弛变量 r1, r2, r3 转换成等式。
"""

# 已知约束条件：
# 1）y + x <= 3        =>  转为等式：y + x + r1² = 3
# 2) x >= 0            =>  转为等式：x - r2² = 0
# 3) y >= 0            =>  转为等式：y - r3² = 0
# 求 3x + 4y - 2 的最大值

import torch


# --- 1. 定义模型 ---
class MyModel(torch.nn.Module):
    """将 x, y 以及松弛变量 r1, r2, r3 作为可训练参数。"""

    def __init__(self):
        super().__init__()
        self.x = torch.nn.Parameter(torch.tensor(0.1))
        self.y = torch.nn.Parameter(torch.tensor(0.1))
        self.r1 = torch.nn.Parameter(torch.tensor(0.1))   # 松弛变量，对应约束 1
        self.r2 = torch.nn.Parameter(torch.tensor(0.1))   # 松弛变量，对应约束 2
        self.r3 = torch.nn.Parameter(torch.tensor(0.1))   # 松弛变量，对应约束 3

    def forward(self):
        # 返回四个值：三个约束等式的残差 + 目标函数值
        return (self.y + self.x + self.r1**2 - 3,      # 约束 1 的残差
                self.x - self.r2**2,                    # 约束 2 的残差
                self.y - self.r3**2,                    # 约束 3 的残差
                3*self.x + 4*self.y - 2)               # 目标函数值（要最大化）


# --- 2. 训练循环 ---
model = MyModel()
# 准备优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.035)

# 训练
model.train()                       # 模型进入训练模式
for epoch in range(18000):          # 训练 18000 轮（约束优化需要更多轮次）
    goal1, goal2, goal3, goal = model()     # 前向传播
    # loss = -目标值 + 惩罚项（约束残差越小越好）
    # 大系数 10000 确保约束优先满足，然后再最大化目标值
    loss = -goal + 10000*(goal1**2 + goal2**2 + goal3**2)
    loss.backward()                 # 反向传播梯度（即根据损失优化模型的参数）
    optimizer.step()                # 优化器根据梯度张量优化模型的参数
    optimizer.zero_grad()           # 清空梯度，否则会影响下一次循环
    epoch += 1
    if epoch % 100 == 0:
        print(f'epoch {epoch:4d}: loss = {loss.item():.6f}')
print('训练完毕')

# --- 3. 输出结果 ---
x = model.x.item()
y = model.y.item()
print(f"x = {x}")
print(f"y = {y}")
