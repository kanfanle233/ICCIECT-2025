"""
用 PyTorch 自动求导逼近 2 的平方根。

教学重点：把待求数看成可训练参数，通过 loss 让参数平方接近 2。
"""

import torch


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self):
        return self.x ** 2 - 2


model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- 训练循环 (推荐写法) ---
for epoch in range(1000):
    optimizer.zero_grad()
    goal = model()
    loss = goal ** 2
    loss.backward()
    optimizer.step()


x = model.x.detach().item()

print(f'模型找到的解 x = {x:.8f}')
print(f'验证: x² 的值是 {x ** 2:.8f}')  # 打印 x² 的值，让结果更清晰
print(f'将解代入原方程 x² - 2，结果为: {x ** 2 - 2:.8f}')  # 验证方程结果是否接近 0