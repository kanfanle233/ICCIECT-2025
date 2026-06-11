#coding=utf-8
#Author:haobo
#Date:2022-4-23

"""
K-Means 聚类示例 —— 西瓜书代码复现

使用 scikit-learn 的 KMeans 对随机二维数据进行聚类。
教学重点：
    1. K-Means 是一种无监督学习算法，将数据划分为 K 个簇
    2. 通过迭代更新质心直到收敛
算法原理：
    K-Means 算法流程：
    1) 随机初始化 K 个质心（本例中 K=2）
    2) 将每个样本分配到距离最近的质心所在的簇
    3) 重新计算每个簇的质心（所有样本的均值）
    4) 重复步骤 2-3 直到质心不再变化（收敛）
"""
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn import datasets


# --- 1. 生成随机数据并可视化（聚类前） ---
X = np.random.rand(100, 2) # 生成 100 个二维随机点
plt.scatter(X[:, 0], X[:, 1], marker='o')

# 初始化我们的质心，从原有的数据中选取K个作为质心
def InitCentroids(X, k):
    """随机从数据中选取 k 个样本作为初始质心"""
    index = np.random.randint(0,len(X)-1,k)
    return X[index]

# --- 2. 执行 K-Means 聚类（聚类后） ---
kmeans = KMeans(n_clusters=2).fit(X) # n_clusters=2：将数据分为 2 个簇
label_pred = kmeans.labels_ # 获取每个样本的簇标签
plt.scatter(X[:, 0], X[:, 1], c=label_pred) # 按簇标签着色
plt.show()