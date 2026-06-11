"""
用两层全连接网络拟合 sin 函数，演示非线性回归。
教学重点：Linear + ReLU 组成非线性拟合网络、训练/测试分离、torch.no_grad 推理模式、matplotlib 预测对比曲线。
"""

# 一元线性回归，即最小二乘法
# 已知函数f:x->y的若干样本{x[i]}和{y[i]}, i=1, 2, ..., n。
# 求函数f(x) = c(relu(ax + b)) + d，使得mean(Sigma( (f(x[i]) - y[i])**2, i=1,...,n))达到最小
# relu(x) = max(0, x)

import torch
import numpy as np
import matplotlib.pyplot as ppl

# ── 设备检测（优先 MPS，其次 CUDA，最后 CPU）──────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("使用 MPS (Apple Silicon GPU) 加速")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("使用 CUDA 加速")
else:
    device = torch.device("cpu")
    print("使用 CPU 训练")

# --- 1. 准备样本 ---
# 准备样本
x = np.random.uniform(-np.pi, np.pi, [500, 1])
print("x.shape=", x.shape)  # [100]
y = np.sin(x)
print("y.shape=", y.shape)  # [500]

# --- 2. 创建模型（两层全连接 + ReLU 激活）---
# 创建模型
class MyModel(torch.nn.Module):
    """两层全连接网络：Linear(1,500) -> ReLU -> Linear(500,1)"""
    def __init__(self):
        super().__init__()
        self.line1 = torch.nn.Linear(1, 500)   # 第一层：1维输入 -> 500维隐藏
        self.relu = torch.nn.ReLU()             # relu(x) == max(0, x) 非线性激活
        self.line2 = torch.nn.Linear(500, 1)   # 第二层：500维隐藏 -> 1维输出

    def forward(self, x):   # [?, 1]
        y = self.line1(x)   # [?, 500]
        y = self.relu(y)    # [?, 500]
        y = self.line2(y)   # [?, 1]
        return y

model = MyModel().to(device)
# 准备优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# --- 3. 训练 ---
# 训练
model.train()               # 模型进入训练模式
x = torch.tensor(x).float().to(device)
y = torch.tensor(y).float().to(device)
for epoch in range(2000):   # 训练若干轮
    goal = model(x)         # 构建预测张量（张量可以理解为公式）
    loss = torch.mean(torch.square(goal - y)) # 损失必须是一个标量(scalar)
    loss.backward()         # 反向传播梯度（即根据损失优化模型的参数）
    optimizer.step()        # 优化器根据梯度张量优化模型的参数
    optimizer.zero_grad()   # 清空梯度。否则会影响下一次循环
    epoch += 1
    if epoch % 100 == 0:
        print(f'epoch {epoch:4d}: loss = {loss.item():.10f}')
print('训练完毕')

# --- 4. 测试与可视化 ---
# 测试
x = np.linspace(-2*np.pi, 2*np.pi, 1000, dtype=np.float32)  # 测试范围扩大到 [-2pi, 2pi]
y = np.sin(x)
model.eval()
with torch.no_grad():
    p = torch.tensor(x.reshape([-1, 1])).to(device)
    p = model(p)
    p = p.cpu().numpy().reshape([-1])
ppl.plot(x, y, color="red", label="sin")
ppl.plot(x, p, color="blue", label="predict")
ppl.show()


