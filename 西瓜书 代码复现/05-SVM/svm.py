"""
支持向量机（SVM）示例 —— 西瓜书代码复现

演示线性核、多项式核、高斯核（RBF）三种 SVM 在二维数据和 MNIST 上的分类效果。
教学重点：
    1. SVM 的核心思想是找到最大间隔超平面将两类数据分开
    2. 核函数将数据映射到高维空间，使原本线性不可分的数据变得可分
    3. 惩罚系数 C 控制误分类的容忍度，gamma 控制高斯核的影响范围
算法原理：
    SVM 求解以下优化问题：
    min 1/2 ||w||^2 + C * sum(xi_i)
    s.t. y_i(w*x_i + b) >= 1 - xi_i, xi_i >= 0
    其中 C 是惩罚系数，xi_i 是松弛变量。
    核技巧：K(x,z) = phi(x) · phi(z)，无需显式计算高维映射。
"""
# 线性svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# --- 1. 构造二维分类数据集 ---
data = np.array([
    [0.1, 0.7],
    [0.3, 0.6],
    [0.4, 0.1],
    [0.5, 0.4],
    [0.8, 0.04],
    [0.42, 0.6],
    [0.9, 0.4],
    [0.6, 0.5],
    [0.7, 0.2],
    [0.7, 0.67],
    [0.27, 0.8],
    [0.5, 0.72]
])# 建立数据集
label = [1] * 6 + [0] * 6 #前六个数据的label为1后六个为0

# 生成用于绘制决策边界的网格点
x_min, x_max = data[:, 0].min() - 0.2, data[:, 0].max() + 0.2
y_min, y_max = data[:, 1].min() - 0.2, data[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.002),
                     np.arange(y_min, y_max, 0.002)) # meshgrid生成二维网格坐标
print(xx)

# --- 2. 线性核 SVM ---
# kernel='linear'：线性核 K(x,z) = x·z，适用于线性可分数据
# C=0.001：较小的 C 值允许更多误分类，得到更宽的间隔
model_linear = svm.SVC(kernel='linear', C = 0.001)# 线性svm
model_linear.fit(data, label) # 训练
Z = model_linear.predict(np.c_[xx.ravel(), yy.ravel()]) # 对网格点预测，用于绘制决策区域
Z = Z.reshape(xx.shape) # 将预测结果恢复为网格形状
plt.contourf(xx, yy, Z, cmap = plt.cm.ocean, alpha=0.6) # 绘制决策区域
plt.scatter(data[:6, 0], data[:6, 1], marker='o', color='r', s=100, lw=3) # 正类样本
plt.scatter(data[6:, 0], data[6:, 1], marker='x', color='k', s=100, lw=3) # 负类样本
plt.title('Linear SVM')
plt.show()

# --- 3. 多项式核 SVM ---
# 多项式核 K(x,z) = (gamma * x·z + coef0)^degree
# degree 越大，决策边界越复杂，容易过拟合
plt.figure(figsize=(16, 15))

for i, degree in enumerate([1, 3, 5, 7, 9, 12]):  # 多项式次数选择了1,3,5,7,9,12
    # C: 惩罚系数，gamma: 高斯核的系数
    model_poly = svm.SVC(C=0.0001, kernel='poly', degree=degree)  # 多项式核
    model_poly.fit(data, label)  # 训练
    # ravel - flatten
    # c_ - vstack
    # 把后面两个压扁之后变成了x1和x2，然后进行判断，得到结果在压缩成一个矩形
    Z = model_poly.predict(np.c_[xx.ravel(), yy.ravel()])  # 预测
    Z = Z.reshape(xx.shape)

    plt.subplot(3, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.contourf(xx, yy, Z, cmap=plt.cm.ocean, alpha=0.6) # 绘制决策区域

    # 画出训练点
    plt.scatter(data[:6, 0], data[:6, 1], marker='o', color='r', s=100, lw=3)
    plt.scatter(data[6:, 0], data[6:, 1], marker='x', color='k', s=100, lw=3)
    plt.title('Poly SVM with $\degree=$' + str(degree))
plt.show()


# --- 4. 高斯核（RBF）SVM ---
# 高斯核 K(x,z) = exp(-gamma * ||x-z||^2)，也称径向基函数核
# gamma 越大，每个样本的影响范围越小，决策边界越不规则（容易过拟合）
plt.figure(figsize=(16, 15))

for i, gamma in enumerate([1, 5, 15, 35, 45, 55]):
    # C: 惩罚系数，gamma: 高斯核的系数
    model_rbf = svm.SVC(kernel='rbf', gamma=gamma, C=0.0001).fit(data, label)

    # ravel - flatten
    # c_ - vstack
    # 把后面两个压扁之后变成了x1和x2，然后进行判断，得到结果在压缩成一个矩形
    Z = model_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.subplot(3, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.contourf(xx, yy, Z, cmap=plt.cm.ocean, alpha=0.6) # 绘制决策区域

    # 画出训练点
    plt.scatter(data[:6, 0], data[:6, 1], marker='o', color='r', s=100, lw=3)
    plt.scatter(data[6:, 0], data[6:, 1], marker='x', color='k', s=100, lw=3)
    plt.title('RBF SVM with $\gamma=$' + str(gamma))
plt.show()

# --- 5. 在 MNIST 数据集上测试不同核函数的 SVM ---
# 测试不同SVM在Mnist数据集上的分类情况
# 添加目录到系统路径方便导入模块，该项目的根目录为".../machine-learning-toy-code"
import sys
from pathlib import Path


from sklearn import svm
import numpy as np
from time import time
from sklearn.metrics import accuracy_score
from struct import unpack
from sklearn.model_selection import GridSearchCV

def readimage(path):
    """读取 MNIST 图像二进制文件，返回 (num_samples, 784) 的 numpy 数组"""
    with open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16)) # 解析文件头：魔数、图片数、行数、列数
        img = np.fromfile(f, dtype=np.uint8).reshape(num, 784) # 读取像素数据并 reshape 为二维
    return img

def readlabel(path):
    """读取 MNIST 标签二进制文件，返回标签数组"""
    with open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8)) # 解析文件头：魔数、标签数
        lab = np.fromfile(f, dtype=np.uint8)
    return lab

# 读取 MNIST 数据（二进制格式）
train_data  = readimage("datasets/MNIST/raw/train-images-idx3-ubyte")#读取数据
train_label = readlabel("datasets/MNIST/raw/train-labels-idx1-ubyte")
test_data   = readimage("datasets/MNIST/raw/t10k-images-idx3-ubyte")
test_label  = readlabel("datasets/MNIST/raw/t10k-labels-idx1-ubyte")
print(train_data.shape)
print(train_label.shape)
#数据集中数据太多，为了节约时间，我们只使用前2000张进行训练
train_data=train_data[:2000]
train_label=train_label[:2000]
test_data=test_data[:200]
test_label=test_label[:200]

# --- 5a. 高斯核 SVM ---
svc=svm.SVC() # 创建 SVM 分类器实例
parameters = {'kernel':['rbf'], 'C':[1]}#使用了高斯核
print("Train...")
clf=GridSearchCV(svc,parameters,n_jobs=-1) # GridSearchCV 自动搜索最优参数
start = time()
clf.fit(train_data, train_label)
end = time()
t = end - start
print('Train：%dmin%.3fsec' % (t//60, t - 60 * (t//60)))#显示训练时间
prediction = clf.predict(test_data)#对测试数据进行预测
print("accuracy: ", accuracy_score(prediction, test_label))
# 逐样本统计各类别准确率
accurate=[0]*10
sumall=[0]*10
i=0
j=0
while i<len(test_label):#计算测试集的准确率
    sumall[test_label[i]]+=1
    if prediction[i]==test_label[i]:
        j+=1
    i+=1
print("测试集准确率：",j/200)

# --- 5b. 多项式核 SVM ---
parameters = {'kernel':['poly'], 'C':[1]}#使用了多项式核
print("Train...")
clf=GridSearchCV(svc,parameters,n_jobs=-1)
start = time()
clf.fit(train_data, train_label)
end = time()
t = end - start
print('Train：%dmin%.3fsec' % (t//60, t - 60 * (t//60)))
prediction = clf.predict(test_data)
print("accuracy: ", accuracy_score(prediction, test_label))
accurate=[0]*10
sumall=[0]*10
i=0
j=0
while i<len(test_label):#计算测试集的准确率
    sumall[test_label[i]]+=1
    if prediction[i]==test_label[i]:
        j+=1
    i+=1
print("测试集准确率：",j/200)

# --- 5c. 线性核 SVM ---
parameters = {'kernel':['linear'], 'C':[1]}#使用了线性核
print("Train...")
clf=GridSearchCV(svc,parameters,n_jobs=-1)
start = time()
clf.fit(train_data, train_label)
end = time()
t = end - start
print('Train：%dmin%.3fsec' % (t//60, t - 60 * (t//60)))
prediction = clf.predict(test_data)
print("accuracy: ", accuracy_score(prediction, test_label))
accurate=[0]*10
sumall=[0]*10
i=0
j=0
while i<len(test_label):#计算测试集的准确率
    sumall[test_label[i]]+=1
    if prediction[i]==test_label[i]:
        j+=1
    i+=1
print("测试集准确率：",j/200)
