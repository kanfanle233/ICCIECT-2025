#coding=utf-8
#Author:haobo
#Date:2022-4-23

"""
主成分分析（PCA）降维示例 —— 西瓜书代码复现

使用 scikit-learn 的 PCA 对三维数据进行降维。
教学重点：
    1. PCA 通过正交变换将原始特征转换为一组线性无关的主成分
    2. 每个主成分按方差大小排序，第一主成分方向方差最大
    3. 通过保留前 k 个主成分实现降维
算法原理：
    PCA 算法步骤：
    1) 对数据进行中心化（减去均值）
    2) 计算协方差矩阵 C = (1/n) * X^T * X
    3) 对协方差矩阵进行特征值分解，得到特征值和特征向量
    4) 按特征值从大到小排序，选取前 k 个特征向量构成投影矩阵
    5) 将原始数据投影到 k 维子空间：X_new = X * W_k
    方差保留比例（explained_variance_ratio_）衡量降维后信息的保留程度。
"""
#首先我们生成随机数据并可视化
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs

# --- 1. 生成三维聚类数据 ---
# X为样本特征，Y为样本簇类别， 共10000个样本，每个样本3个特征，共4个簇
X, y = make_blobs(n_samples=10000, n_features=3, centers=[[3,3, 3], [0,0,0], [1,1,1], [2,2,2]], cluster_std=[0.2, 0.1, 0.2, 0.2], random_state =9)
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
plt.scatter(X[:, 0], X[:, 1], X[:, 2],marker='o') # 三维散点图

# --- 2. 保留所有主成分，查看各主成分方差占比 ---
#我们先不降维，只对数据进行投影，看看投影后的三个维度的方差分布，代码如下：
from sklearn.decomposition import PCA
pca = PCA(n_components=3) # 保留全部 3 个主成分
pca.fit(X)
print(pca.explained_variance_ratio_) # 各主成分方差占总方差的比例
print(pca.explained_variance_)       # 各主成分的方差值

# --- 3. 降维：从 3 维降到 2 维 ---
#现在我们来进行降维，从三维降到2维，代码如下：
pca = PCA(n_components=2) # 只保留前 2 个主成分
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)

# --- 4. 可视化降维后的数据 ---
#为了有个直观的认识，我们看看此时转化后的数据分布，代码如下：
X_new = pca.transform(X) # 将 3D 数据投影到 2D
plt.scatter(X_new[:, 0], X_new[:, 1],marker='o')
plt.show()

# --- 5. 按方差保留比例自动选择主成分数 ---
#现在我们看看不直接指定降维的维度，而指定降维后的主成分方差和比例。
print('n_components=0.95') # 保留 95% 方差所需的最少主成分数
pca = PCA(n_components=0.95)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_) # 自动选择的主成分数

print('n_components=0.99') # 保留 99% 方差
pca = PCA(n_components=0.99)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_)

print('n_components=mle') # 使用 MLE（最大似然估计）自动选择主成分数
pca = PCA(n_components='mle')
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_)