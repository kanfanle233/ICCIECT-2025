"""
用 PyTorch 优化方法解简单方程组。

教学重点：把未知数当作可训练参数，让方程误差逐步变小。
模型通过最小化目标函数 (x-2)^2 + (y-3)^2 来逼近方程组的解。
"""

import torch


# --- 1. 定义模型 ---
class MyModel(torch.nn.Module):
    """将 x 和 y 作为可训练参数，通过优化找到方程组的解。"""

    def __init__(self):
        super().__init__()
        # 将 x 和 y 的初始值设置为 0.1
        self.x = torch.nn.Parameter(torch.tensor(0.1))
        self.y = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self):
        # 示例目标函数: (x - 2)^2 + (y - 3)^2，最小值在 x=2, y=3 处
        return (self.x - 2)**2 + (self.y - 3)**2


# --- 2. 训练循环 ---
# 实例化模型和优化器
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(2000):
    loss = model()           # 前向传播，计算目标函数值
    loss.backward()          # 反向传播，计算梯度
    optimizer.step()         # 根据梯度更新参数
    optimizer.zero_grad()    # 清空梯度，为下一轮做准备

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1:4d}: loss = {loss.item():.6f}')

print('训练完毕')

# --- 3. 输出结果 ---
# detach() 把张量从计算图中分离出来，.item() 取出标量值
x = model.x.detach().item()
y = model.y.detach().item()
print(f"x = {x:.6f}")
print(f"y = {y:.6f}")

# 验证解是否满足方程：越接近 0 表示方程成立
print("x + 3y - 4 =", x + 3 * y - 4)
print("4x - 2y - 1 =", 4 * x - 2 * y - 1)
