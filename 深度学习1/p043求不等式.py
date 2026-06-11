"""
用 PyTorch 处理不等式约束示例。

教学重点：把违反约束的部分设计成惩罚项。
当方程有多个解时，初始值的不同会导致模型收敛到不同的解。
"""

import torch


# --- 1. 定义模型 ---
# 目标：求解一元二次方程 x² - 3x - 4 = 0
# (这个方程的解是 x = -1 和 x = 4)

class MyModel(torch.nn.Module):
    """将 x 作为可训练参数，通过优化逼近方程的解。"""

    def __init__(self, initial_x):
        super().__init__()
        # 使用传入的初始值
        self.x = torch.nn.Parameter(torch.tensor(initial_x, dtype=torch.float32))

    def forward(self):
        # forward 函数计算方程的"误差"
        # 误差 = x² - 3x - 4
        return self.x ** 2 - 3 * self.x - 4


# --- 2. 实验一：寻找解 x = 4 ---
# 初始值 10.0 远离 x=-1，更靠近 x=4，所以会收敛到 x=4
print("--- 实验一：从 x = 10.0 开始 ---")
model_1 = MyModel(initial_x=10.0)
optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=0.1)  # 使用稍大的学习率

model_1.train()
for epoch in range(500):
    optimizer_1.zero_grad()        # 清空梯度
    error = model_1()              # 前向传播，计算方程残差
    loss = error ** 2              # 损失 = 残差的平方，最小值为 0
    loss.backward()                # 反向传播梯度
    optimizer_1.step()             # 更新参数

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1:4d}: Loss = {loss.item():.8f}, x = {model_1.x.item():.4f}')

# --- 3. 输出结果 ---
print('\n--- 训练完毕 ---')
final_x_1 = model_1.x.detach().item()
print(f"模型找到的解是: x = {final_x_1:.4f}")
print(f"验证方程 (x² - 3x - 4): {(final_x_1 ** 2 - 3 * final_x_1 - 4):.8f} (目标是 0.0)")