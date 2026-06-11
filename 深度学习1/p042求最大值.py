"""
用 PyTorch 求函数最大值。

教学重点：最大化问题可以转换为最小化负数 loss。
"""

import torch


# 目标：求解一元二次方程 x² - 3x - 4 = 0
# (这个方程的解是 x = -1 和 x = 4)

class MyModel(torch.nn.Module):
    def __init__(self, initial_x):
        super().__init__()
        # 使用传入的初始值
        self.x = torch.nn.Parameter(torch.tensor(initial_x, dtype=torch.float32))

    def forward(self):
        # forward 函数计算方程的“误差”
        # 误差 = x² - 3x - 4
        return self.x ** 2 - 3 * self.x - 4


# --- 实验一：寻找解 x = 4 ---
print("--- 实验一：从 x = 10.0 开始 ---")
model_1 = MyModel(initial_x=10.0)
optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=0.1)  # 使用稍大的学习率

model_1.train()
for epoch in range(500):
    optimizer_1.zero_grad()
    error = model_1()
    loss = error ** 2
    loss.backward()
    optimizer_1.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1:4d}: Loss = {loss.item():.8f}, x = {model_1.x.item():.4f}')

print('\n--- 训练完毕 ---')
final_x_1 = model_1.x.detach().item()
print(f"模型找到的解是: x = {final_x_1:.4f}")
print(f"验证方程 (x² - 3x - 4): {(final_x_1 ** 2 - 3 * final_x_1 - 4):.8f} (目标是 0.0)")