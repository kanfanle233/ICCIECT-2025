"""
利用 PyTorch 自动微分求任意正数的平方根。
教学重点：nn.Parameter 定义可学习参数、loss 构建、backward 梯度计算、optimizer 更新参数的完整训练流程。
"""

import torch

class MyModel(torch.nn.Module):
    """平方根模型：通过学习 x 使得 x^2 - a = 0，即 x = sqrt(a)"""
    def __init__(self):
        super().__init__()
        self.x = torch.nn.Parameter(torch.tensor(1.0))  # 可学习参数，初始值为 1.0

    def forward(self, a):
        return self.x**2 - a  # 当 x^2 = a 时，输出为 0（即 loss 最小）

# --- 1. 创建模型与优化器 ---
model = MyModel()
# 准备优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- 2. 训练：通过最小化 (x^2 - a)^2 来求 a 的平方根 ---
# 训练
model.train()               # 模型进入训练模式
a = 3     # 求a的平方根
for epoch in range(200):      # 训练200轮
    pred = model(a)         # 构建预测张量（张量可以理解为公式）
    loss = torch.square(pred) # 构建损失张量
    loss.backward()         # 反向传播梯度（即根据损失优化模型的参数）
    optimizer.step()        # 优化器根据梯度张量优化模型的参数
    optimizer.zero_grad()   # 清空梯度。否则会影响下一次循环
    epoch += 1
    if epoch % 100 == 0:
        print(f'epoch {epoch:4d}: loss = {loss.item():.6f}')
print('训练完毕')

# --- 3. 输出结果 ---
# 输出平方根
print(model.x.detach().item())  # detach() 脱离计算图，item() 取标量值
