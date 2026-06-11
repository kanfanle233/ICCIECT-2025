"""
多元线性回归示例，使用 torch.nn.Linear 拟合多输入单输出函数。
教学重点：nn.Linear 线性层定义、numpy 到 Tensor 的转换、训练循环（前向传播、损失计算、反向传播、参数更新）、损失曲线绘制。
"""

# --- 1. 问题描述 ---
# 多元线性回归, 使用torch.nn.Linear
# 已知函数f:X->y的若干样本{X[i, :]}和{y[i]}, i=1, 2, ..., n。
# 求函数f(x) = XA+b，使得mean(Sigma( (f(X[i]) - y[i])**2, i=1,...,n))达到最小
#
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

# --- 2. 准备样本 ---
# 准备样本, 4元线性回归
x = np.random.uniform(-2, 2, [500, 4])
print("x=\n", x)
print("x.shape=", x.shape)  # [500]
# 计算y = xA + b，其中：
#   1）x.ndim == 2 and x.shape[1] == A.shape[0]
#   2）A.ndim == 2
#   3）b.ndim == 1 and b.shape[0] == A.shape[1]
#   4) y.ndim == 2 and y.shape == (x.shape[0], A.shape[1])
y = np.matmul(x, [[2], [-1], [3], [-2]]) + [2]

# 增加噪声。实际应用中样本数据不会那么精准，总会有点噪声
y += np.random.normal(0, 0.1, [500, 1])
print("y=\n", x)
print("y.shape=", y.shape)  # [500]

# --- 3. 创建模型 ---
# 创建模型
class MyModel(torch.nn.Module):
    """使用 torch.nn.Linear 的线性回归模型：y = XA + b"""
    def __init__(self):
        super().__init__()
        # 线性变换预定义模型
        self.line = torch.nn.Linear(4, 1)

    def forward(self, x):   # x: [500, 4]
        return self.line(x) # [500,1]

model = MyModel().to(device)
# 准备优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- 4. 训练 ---
# 训练
model.train()               # 模型进入训练模式
x = torch.tensor(np.float32(x)).to(device)
y = torch.tensor(np.float32(y)).to(device)
losses = []
for epoch in range(2000):   # 训练若干轮
    goal = model(x)         # [?, 1]
    loss = torch.mean(torch.square(goal - y)) # 损失必须是一个标量(scalar)
    loss.backward()         # 反向传播梯度（即根据损失优化模型的参数）
    optimizer.step()        # 优化器根据梯度张量优化模型的参数
    optimizer.zero_grad()   # 清空梯度。否则会影响下一次循环
    losses.append(loss.item())
    epoch += 1
    if epoch % 100 == 0:
        print(f'epoch {epoch:4d}: loss = {loss.item():.10f}')
print('训练完毕')

model.eval()
with torch.no_grad():
    for p in model.parameters():
        print(p.cpu().numpy())

ppl.plot(np.arange(len(losses)), losses)
ppl.show()

