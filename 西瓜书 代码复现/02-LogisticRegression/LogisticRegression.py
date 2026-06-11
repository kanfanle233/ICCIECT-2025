"""
Logistic 回归（逻辑回归）示例 —— 西瓜书代码复现

使用 scikit-learn 的 LogisticRegression 对 MNIST 手写数字数据集进行分类。
教学重点：
    1. Logistic 回归通过 Sigmoid 函数将线性组合映射到 [0,1] 概率区间
    2. L1 正则化（Lasso）可产生稀疏权重，实现特征选择
    3. saga 求解器适用于大规模数据集和 L1 正则化
算法原理：
    Logistic 回归虽然名为"回归"，实际上是一种分类算法。
    它对每个类别学习一组权重 w 和偏置 b，计算 P(y=k|x) = softmax(w*x + b)。
    训练时通过极大似然估计（等价于最小化交叉熵损失）优化参数。
"""
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression


# --- 1. 加载 MNIST 数据集 ---
# MNIST 包含 70000 张 28x28 的手写数字灰度图，展平后每张图 784 维
mnist = fetch_openml('mnist_784')
X, y = mnist['data'], mnist['target']
# 前 60000 张作为训练集，后 10000 张作为测试集
X_train = np.array(X[:60000], dtype=float)
y_train = np.array(y[:60000], dtype=float)
X_test = np.array(X[60000:], dtype=float)
y_test = np.array(y[60000:], dtype=float)

print(X_train.shape) # (60000, 784)
print(y_train.shape) # (60000,)
print(X_test.shape)  # (10000, 784)
print(y_test.shape)  # (10000,)

# --- 2. 训练 Logistic 回归模型 ---
# penalty="l1"：使用 L1 正则化，可将部分权重压缩为 0（稀疏化）
# solver="saga"：适用于大规模数据和 L1 正则化的优化算法
# tol=0.1：容忍度，提前停止条件，值越大训练越快但精度可能降低
clf = LogisticRegression(penalty="l1", solver="saga", tol=0.1)
clf.fit(X_train, y_train) # 训练模型
score = clf.score(X_test, y_test) # 在测试集上评估准确率
print("Test score with L1 penalty: %.4f" % score)