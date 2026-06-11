"""
多层感知机（MLP）示例 —— 西瓜书代码复现

使用 scikit-learn 的 MLPClassifier（多层感知机分类器）对 MNIST 手写数字进行分类。
教学重点：
    1. MLP 是一种前馈神经网络，由输入层、隐藏层和输出层组成
    2. 通过反向传播算法（BP）更新权重，最小化损失函数
    3. alpha 参数控制 L2 正则化强度，防止过拟合
算法原理：
    MLP 每一层的神经元对上一层输出进行线性变换后通过激活函数（如 ReLU）映射。
    输出层使用 softmax 将结果转化为各类别的概率分布。
    本例使用两层隐藏层，每层 15 个神经元。
"""
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
import numpy as np

# --- 1. 加载 MNIST 数据集 ---
# MNIST：70000 张 28x28 手写数字图片，展平后 784 维向量
mnist = fetch_openml('mnist_784')
X, y = mnist['data'], mnist['target']
# 划分训练集（前 60000 张）和测试集（后 10000 张）
X_train = np.array(X[:60000], dtype=float)
y_train = np.array(y[:60000], dtype=float)
X_test = np.array(X[60000:], dtype=float)
y_test = np.array(y[60000:], dtype=float)

# --- 2. 构建并训练 MLP 模型 ---
# alpha=1e-5：L2 正则化系数，防止权重过大导致过拟合
# hidden_layer_sizes=(15,15)：两层隐藏层，每层 15 个神经元
clf = MLPClassifier(alpha=1e-5,
                    hidden_layer_sizes=(15,15), random_state=1)

clf.fit(X_train, y_train) # 使用反向传播算法训练网络

# --- 3. 评估模型 ---
score = clf.score(X_test, y_test) # 在测试集上计算分类准确率
print(f"MLP 测试集准确率: {score:.4f}")


