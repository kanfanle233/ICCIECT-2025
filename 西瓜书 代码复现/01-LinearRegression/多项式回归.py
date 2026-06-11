"""
多项式回归示例 —— 西瓜书代码复现

通过 Pipeline 将多项式特征扩展与线性回归串联，演示欠拟合与过拟合现象。
教学重点：
    1. PolynomialFeatures 将原始特征 x 扩展为 [x, x^2, ..., x^d]
    2. Pipeline 串联特征工程与模型训练
    3. 对比不同多项式次数（degree=1,4,15）的拟合效果
    4. 使用 10 折交叉验证评估均方误差（MSE）
算法原理：
    多项式回归本质上仍是线性回归，只是将特征空间从 x 映射到
    多项式空间 [x, x^2, ..., x^d]，再在高维空间中做线性拟合。
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures # 导入能够计算多项式特征的类
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

def true_fun(X): # 这是我们设定的真实函数，即ground truth的模型
    """真实函数：y = cos(1.5 * pi * x)（非线性关系，用于演示拟合效果）"""
    return np.cos(1.5 * np.pi * X)

# --- 1. 生成带噪声的训练数据 ---
np.random.seed(0)
n_samples = 30 # 设置随机种子

X = np.sort(np.random.rand(n_samples)) # 在 [0,1) 区间均匀采样
y = true_fun(X) + np.random.randn(n_samples) * 0.1 # 真实函数 + 高斯噪声

# --- 2. 对比不同多项式次数的拟合效果 ---
degrees = [1, 4, 15] # 多项式最高次：1 欠拟合，4 合适，15 过拟合
plt.figure(figsize=(14, 5))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=()) # 隐藏坐标轴刻度，突出曲线形状
    # 构造多项式特征：将 x 映射为 [x, x^2, ..., x^degree]
    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)
    linear_regression = LinearRegression()
    # 使用 Pipeline 串联特征扩展与线性回归，形成多项式回归模型
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)]) # 使用pipline串联模型
    pipeline.fit(X[:, np.newaxis], y)

    # 10 折交叉验证，返回负 MSE（sklearn 约定：越大越好，故取负）
    scores = cross_val_score(pipeline, X[:, np.newaxis], y,scoring="neg_mean_squared_error", cv=10) # 使用交叉验证
    X_test = np.linspace(0, 1, 100)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model") # 模型拟合曲线
    plt.plot(X_test, true_fun(X_test), label="True function") # 真实函数曲线
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples") # 训练数据散点
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        degrees[i], -scores.mean(), scores.std())) # 标注 MSE 均值和标准差
plt.show()