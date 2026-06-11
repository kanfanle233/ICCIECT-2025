"""
用 PyTorch 自动求导逼近任意正数的平方根。

教学重点：模型参数、损失函数、优化器三者如何一起工作。
"""

import torch
import math


# 1. 定义一个通用的平方根求解模型
class SqrtSolver(torch.nn.Module):
    def __init__(self, target_number, initial_guess=1.0):
        """
        构造函数，现在可以接收任何目标数字
        :param target_number: 我们要求解其平方根的数字
        :param initial_guess: x 的初始猜测值
        """
        super().__init__()
        # 将目标数字存为一个固定的 tensor
        self.target = torch.tensor(target_number, dtype=torch.float32)
        # x 依然是我们需要学习的参数
        self.x = torch.nn.Parameter(torch.tensor(initial_guess, dtype=torch.float32))

    def forward(self):
        # 核心方程变为 x² - target = 0
        return self.x ** 2 - self.target


# 2. 获取并验证用户输入
target_number = None
while True:
    try:
        # 提示用户输入
        user_input = input("请输入一个非负数来计算其平方根 (输入 q 退出): ")

        # 检查是否退出
        if user_input.lower() == 'q':
            exit()

        # 尝试将输入转换为浮点数
        num = float(user_input)

        # 检查是否为负数
        if num < 0:
            print("错误：无法计算负数的实数平方根，请重新输入。")
            continue

        target_number = num
        break  # 输入有效，跳出循环

    except ValueError:
        # 如果转换失败（例如输入了文本），则提示错误
        print("错误：输入无效，请输入一个数字。")

# 3. 创建模型和优化器
# 使用用户输入的数字来初始化模型
model = SqrtSolver(target_number=target_number)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. 训练循环
print(f"\n--- 正在使用梯度下降法计算 {target_number} 的平方根 ---")
# 对于非常大或非常小的数，可能需要更多次迭代
num_epochs = 2000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model()
    loss = output ** 2
    loss.backward()
    optimizer.step()


# 5. 输出最终结果
print("\n--- 计算完成 ---")
found_sqrt = model.x.detach().item()
actual_sqrt = math.sqrt(target_number)

print(f"您输入的数字是: {target_number}")
print(f"模型计算出的平方根是: {found_sqrt:.8f}")
print(f"Python `math.sqrt` 的精确结果是: {actual_sqrt:.8f}")
print(f"两者误差: {abs(found_sqrt - actual_sqrt):.10f}")
