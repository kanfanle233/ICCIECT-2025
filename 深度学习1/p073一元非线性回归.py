"""
一元非线性回归示例。

教学重点：加入非线性层后，模型可以拟合曲线而不只是直线。
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import numpy as np
import matplotlib.pyplot as ppl

# 1. 准备样本
x_np = np.random.uniform(-np.pi, np.pi, size=[500, 1])
y_np = np.sin(x_np)

print("x.shape=", x_np.shape)
print("y.shape=", y_np.shape)


# 2. 创建模型
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.line1 = torch.nn.Linear(in_features=1, out_features=500)
        self.relu = torch.nn.ReLU()
        self.line2 = torch.nn.Linear(in_features=500, out_features=1)

    def forward(self, x):
        y = self.line1(x)
        y = self.relu(y)
        y = self.line2(y)
        return y


model = MyModel()

# 3. 准备优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. 训练
model.train()  # 模型进入训练模式

# 数据转换
x = torch.from_numpy(x_np).float()
y = torch.from_numpy(y_np).float()

for epoch in range(2000):  # 训练若干轮
    goal = model(x)  # 构建预测张量
    loss = torch.mean(torch.square(goal - y))  # 损失必须是一个标量

    loss.backward()  # 反向传播梯度
    optimizer.step()  # 优化器根据梯度更新参数
    optimizer.zero_grad()  # 清空梯度

    # 注意: for 循环会自动递增 epoch，所以 'epoch += 1' 是不必要的，这里将其省略。

    if (epoch + 1) % 100 == 0:
        print(f'epoch {(epoch + 1):4d}: loss = {loss.item():.10f}')

# 5. 测试与可视化
# 创建新的测试数据
x_test = np.linspace(-2 * np.pi, 2 * np.pi, num=1000, dtype=np.float32)
y_test = np.sin(x_test)

model.eval()  # 模型进入评估模式
with torch.no_grad():  # 禁用梯度计算
    # 准备输入模型的 tensor
    p = torch.from_numpy(x_test.reshape([-1, 1]))
    # 得到预测结果
    p = model(p)
    # 转回 numpy 用于绘图
    p = p.numpy().reshape([-1])

# 绘图
ppl.plot(x_test, y_test, color="red", label="sin")
ppl.plot(x_test, p, color="blue", label="predict")
ppl.legend()
ppl.show()
