"""
朴素贝叶斯分类示例 —— 西瓜书代码复现

使用 scikit-learn 的 GaussianNB（高斯朴素贝叶斯）对二维数据进行分类。
教学重点：
    1. 朴素贝叶斯基于贝叶斯定理和"特征条件独立"假设
    2. 高斯朴素贝叶斯假设每个特征在每个类别下服从正态分布
    3. predict_proba 返回样本属于每个类别的后验概率
算法原理：
    贝叶斯定理：P(y|x) = P(x|y) * P(y) / P(x)
    "朴素"假设：P(x1,x2,...,xn|y) = prod(P(xi|y))，即特征之间相互独立
    预测时选择后验概率最大的类别：y* = argmax_y P(y) * prod(P(xi|y))
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs

# --- 1. 生成二维分类数据 ---
#生成随机数据
# make_blobs：为聚类产生数据集
# n_samples：样本点数，n_features：数据的维度，centers:产生数据的中心点，默认值3
# cluster_std：数据集的标准差，浮点数或者浮点数序列，默认值1.0，random_state：随机种子
X, y = make_blobs(n_samples = 100, n_features=2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
plt.show()

# --- 2. 训练高斯朴素贝叶斯模型 ---
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()#朴素贝叶斯
model.fit(X, y)# 训练模型：计算每个类别下每个特征的均值和方差

# --- 3. 对测试数据进行预测并可视化 ---
rng = np.random.RandomState(0)
X_test = [-6, -14] + [14, 18] * rng.rand(2000, 2)#生成测试集：在较大范围内随机生成数据
y_pred = model.predict(X_test)
# 将训练集和测试集的数据用图像表示出来，颜色深直径大的为训练集，颜色浅直径小的为测试集
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu') # 训练数据（大点）
lim = plt.axis()
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, s=20, cmap='RdBu', alpha=0.1) # 测试数据（小点，半透明）
plt.axis(lim)
plt.show()

# --- 4. 查看预测概率 ---
yprob = model.predict_proba(X_test)#返回的预测值为，每条数据对每个分类的概率
print(yprob[-8:].round(2)) # 打印最后 8 个样本的类别概率