"""
用 PyTorch 全连接网络对 make_moons 数据集进行二分类。
教学重点：BinaryClassifier 模型构建（Linear + ReLU + Sigmoid）、BCELoss 二分类交叉熵、决策边界可视化。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --- 1. 设备检测 ---
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

# --- 2. 准备数据 ---
# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 1. 创建合成数据集（1000个样本，噪声标准差0.2）
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

# 将数据转换为PyTorch张量并送入设备
X = torch.from_numpy(X).float().to(device)
y = torch.from_numpy(y).float().to(device)

# 重塑y的形状为(n_samples, 1)，与模型输出维度匹配
y = y.view(-1, 1)

# 划分训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 3. 定义神经网络模型 ---
# 2. 定义神经网络模型
class BinaryClassifier(nn.Module):
    """二分类器：输入层 -> 隐藏层(ReLU) -> 输出层(Sigmoid)"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)  # 第一层：输入到隐藏
        self.relu = nn.ReLU()                             # ReLU 激活函数
        self.layer2 = nn.Linear(hidden_dim, output_dim)  # 第二层：隐藏到输出
        self.sigmoid = nn.Sigmoid()  # y = 1/(1 + e**(-x))，将输出映射到 [0,1]

    def forward(self, x):   # x: [?, input_dim]
        x = self.layer1(x)  # [?, hidden_dim]
        x = self.relu(x)    # 非线性激活
        x = self.layer2(x)  # [?, output_dim]
        x = self.sigmoid(x) # 输出概率值 [0, 1]
        return x


# 初始化模型（2维输入 -> 10个隐藏神经元 -> 1维输出）
input_dim = 2
hidden_dim = 10
output_dim = 1
model = BinaryClassifier(input_dim, hidden_dim, output_dim).to(device)

# --- 4. 训练模型 ---
# 3. 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失（Binary Cross Entropy）
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 1000
losses = []

for epoch in range(num_epochs):
    # 前向传播：计算预测值和损失
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # 反向传播和优化
    optimizer.zero_grad()   # 清空梯度
    loss.backward()         # 计算梯度
    optimizer.step()        # 更新参数

    losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# --- 5. 评估模型 ---
# 5. 评估模型
model.eval()  # 设置模型为评估模式（关闭 Dropout 等训练专用操作）
with torch.no_grad():  # 推理时不需要计算梯度
    # 训练集准确率
    train_outputs = model(X_train)
    train_predicted = (train_outputs >= 0.5).float()  # 概率 >= 0.5 判为正类
    train_accuracy = accuracy_score(y_train.cpu(), train_predicted.cpu())

    # 测试集准确率
    test_outputs = model(X_test)
    test_predicted = (test_outputs >= 0.5).float()
    test_accuracy = accuracy_score(y_test.cpu(), test_predicted.cpu())

print(f'训练集准确率: {train_accuracy:.4f}')
print(f'测试集准确率: {test_accuracy:.4f}')


# --- 6. 可视化 ---
# 6. 可视化决策边界（可选）
def plot_decision_boundary(model, X, y):
    """绘制模型的决策边界"""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01  # 网格步长
    # 生成网格点
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 对网格中每个点预测类别
    Z = model(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float().to(device))
    Z = (Z >= 0.5).float().cpu().numpy().reshape(xx.shape)

    # 绘制决策区域和数据点
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50)
    plt.title("决策边界")
    plt.show()

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 绘制决策边界
plot_decision_boundary(model, X.cpu().numpy(), y.cpu().numpy())

# 绘制损失曲线
plt.plot(losses)
plt.title('训练损失曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
