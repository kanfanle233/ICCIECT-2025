#coding=utf-8
#Author:haobo
#Date:2022-4-23

"""
k 近邻（kNN）分类示例 —— 西瓜书代码复现

使用 scikit-learn 的 KNeighborsClassifier 对手写数字数据集进行分类。
教学重点：
    1. kNN 是一种"懒惰学习"算法，训练时不建立模型，预测时才计算
    2. 对于待分类样本，找到训练集中距离最近的 K 个邻居
    3. 通过多数投票决定该样本的类别
算法原理：
    kNN 算法步骤：
    1) 计算待预测样本与所有训练样本的距离（默认欧氏距离）
    2) 选取距离最小的 K 个训练样本（K 个邻居）
    3) 在这 K 个邻居中，出现次数最多的类别即为预测结果
    默认参数：K=5，距离度量=欧氏距离，权重=uniform（等权）
"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# --- 1. 加载手写数字数据集 ---
# load_digits：1797 张 8x8 手写数字图片，展平后 64 维
digits = load_digits()
data = digits.data     # 特征集：(1797, 64)
target = digits.target # 目标集：数字 0-9


# --- 2. 划分训练集和测试集 ---
#将数据集拆分为训练集（75%）和测试集（25%）:
train_x, test_x, train_y, test_y = train_test_split(
    data, target, test_size=0.25, random_state=33)


# --- 3. 构建并训练 kNN 分类器 ---
#构造KNN分类器：采用默认参数（K=5，欧氏距离，等权投票）
knn = KNeighborsClassifier()


#拟合模型：kNN 的"训练"只是存储训练数据
knn.fit(train_x, train_y)
#预测数据：对每个测试样本找最近的 5 个邻居，投票决定类别
predict_y = knn.predict(test_x)


# --- 4. 评估模型 ---
#计算模型准确度
score = accuracy_score(test_y, predict_y)
print(score)