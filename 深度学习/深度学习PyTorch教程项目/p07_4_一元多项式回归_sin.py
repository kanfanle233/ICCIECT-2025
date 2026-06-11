"""
用多项式特征 + 线性层拟合 sin 函数，演示多项式回归。
教学重点：PolyModel 自定义多项式特征生成、torch.concat 特征拼接、MSELoss 损失函数使用。
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

# --- 2. 创建模型（多项式特征 + 线性层）---
# 创建模型
class PolyModel(torch.nn.Module):
    """多项式回归模型：将输入 x 转为 [x, x^2/2!, x^3/3!, ...] 多项式特征，再线性拟合"""
    def __init__(self, rank:int):
        super().__init__()
        self.rank = rank                 # 多项式阶数
        self.line = torch.nn.Linear(rank, 1)  # 线性层：rank个多项式特征 -> 1个输出

    def forward(self, x):   # [?, 1]
        poly = []
        power = 1
        for i in range(self.rank):
            power = (power * x)/(i+1)       # [?, 1]
            poly.append(power)
        poly = torch.concat(poly, 1)    # [?, rank]
        y = self.line(poly)
        return y        # [?, 1]

# --- 3. 训练 ---
model = PolyModel(30).to(device)  # 使用30阶多项式
# 准备优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# 训练
model.train()               # 模型进入训练模式
x = torch.tensor(x).float().to(device)
y = torch.tensor(y).float().to(device)
loss_fn = torch.nn.MSELoss()    # 均方差损失
for epoch in range(10000):   # 训练若干轮
    goal = model(x)         # 构建预测张量（张量可以理解为公式）
    # loss = torch.mean(torch.square(goal - y)) # 损失必须是一个标量(scalar)
    loss = loss_fn(goal, y)
    loss.backward()         # 反向传播梯度（即根据损失优化模型的参数）
    optimizer.step()        # 优化器根据梯度张量优化模型的参数
    optimizer.zero_grad()   # 清空梯度。否则会影响下一次循环
    epoch += 1
    if epoch % 100 == 0:
        print(f'epoch {epoch:4d}: loss = {loss.item():.10f}')
print('训练完毕')

# --- 4. 测试与可视化 ---
# 测试
x = np.linspace(-np.pi, np.pi, 1000, dtype=np.float32)
y = np.sin(x)
model.eval()
with torch.no_grad():
    ps = 0
    for p in model.parameters():
        ps += p.numel()
        print(p.cpu().numpy())
    print("parameters:", ps)
    p = torch.tensor(x.reshape([-1, 1])).to(device)
    p = model(p)
    p = p.cpu().numpy().reshape([-1])
ppl.plot(x, y, color="red", label="sin")
ppl.plot(x, p, color="blue", label="predict")
ppl.show()


