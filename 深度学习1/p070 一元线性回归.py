"""
一元线性回归示例。

教学重点：用一条直线学习 x 和 y 的关系。
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 在这里 import 你的库 ---
import torch
import numpy as np
import matplotlib.pyplot as ppl

# --- 1. 数据准备 (使用 _np 后缀表示 NumPy 数组) ---
x_np = np.random.uniform(-2, 2, [500])
# (移除了冗余的 print)
y_np = 3 * x_np + 2
y_np += np.random.normal(0, 0.1, [len(x_np)])

# (移除了冗余的 print)
ppl.scatter(x_np, y_np, alpha=0.5, s=10, label="Original Data")
ppl.title("Original Data")
ppl.legend()
ppl.show()


# --- 2. 模型定义 ---
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化参数 a 和 b
        self.a = torch.nn.Parameter(torch.tensor(0.1))  # 初始猜测 a=0.1
        self.b = torch.nn.Parameter(torch.tensor(0.0))  # 初始猜测 b=0.0

    def forward(self, x):
        # 定义前向传播：y = a*x + b
        return self.a * x + self.b


# --- 3. 初始化和数据转换 ---
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()  # 将模型设置为训练模式

# (修正) 创建新的 Tensor 变量，而不是覆盖 NumPy 变量
x_tensor = torch.tensor(np.float32(x_np))
y_tensor = torch.tensor(np.float32(y_np))

losses = []

# --- 4. 训练循环 ---
for epoch in range(2000):
    goal = model(x_tensor)
    loss = torch.mean(torch.square(goal - y_tensor))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # (修正 1) .item() 是一个方法，需要加括号
    losses.append(loss.item())

    # (修正 2) 删除多余的 epoch += 1

    if epoch % 100 == 0:
        # 这里的 epoch 会正确地显示 0, 100, 200...
        print(f'epoch {epoch:4d}: loss = {loss.item():.10f}')

print("训练完毕")

a = model.a.item()
b = model.b.item()

print(f"真实参数: a=3.0, b=2.0")
print(f"学习到的参数: a={a:.4f}, b={b:.4f}")

# =======================================================
#               --- 补全的代码从这里开始 ---
# =======================================================

# --- 5. 可视化训练结果 ---

# 使用学习到的 a 和 b 来计算模型在原始 x_np 上的预测值
y_pred_np = a * x_np + b

# 创建一个新的图像
ppl.figure(figsize=(10, 5))

# 绘制原始数据点（半透明）
ppl.scatter(x_np, y_np, label='Original Data', alpha=0.3, s=10)

# 绘制模型学习到的直线
ppl.plot(x_np, y_pred_np, color='red', linewidth=2, label=f'Learned Line: y = {a:.2f}x + {b:.2f}')

ppl.title("Linear Regression Result")
ppl.xlabel("x")
ppl.ylabel("y")
ppl.legend()
ppl.show()

# --- 6. 可视化损失函数 (Loss Curve) ---
# 这一步是检查模型训练过程是否“收敛”

# 创建另一个新图像
ppl.figure(figsize=(10, 5))
ppl.plot(losses)
ppl.title("Training Loss Curve")
ppl.xlabel("Epoch")
ppl.ylabel("Loss (MSE)")
ppl.grid(True)
# 由于损失在初期下降很快，使用 log 尺度可以看得更清楚
ppl.yscale('log')
ppl.show()