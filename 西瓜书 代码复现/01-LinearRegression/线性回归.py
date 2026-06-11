"""
线性回归示例 —— 西瓜书代码复现

使用 scikit-learn 的 LinearRegression 对一元线性数据进行拟合。
教学重点：
    1. 线性回归模型 y = w*x + b 的建立与训练
    2. 查看模型参数 w（权重/系数）和 b（偏置/截距）
    3. 将模型预测曲线与真实函数对比可视化
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # 导入线性回归模型


def true_fun(X): # 这是我们设定的真实函数，即ground truth的模型
    """真实函数：y = 1.5x + 0.2（模拟线性关系）"""
    return 1.5*X + 0.2

# --- 1. 生成训练数据 ---
np.random.seed(0) # 设置随机种子，保证结果可复现
n_samples = 30 # 设置采样数据点的个数

'''生成随机数据作为训练集，并且加一些噪声'''
X_train = np.sort(np.random.rand(n_samples)) # 在 [0,1) 区间均匀采样并排序
y_train = (true_fun(X_train) + np.random.randn(n_samples) * 0.05).reshape(n_samples,1) # 加入高斯噪声模拟真实观测

# --- 2. 训练线性回归模型 ---
model = LinearRegression() # 定义模型
model.fit(X_train[:,np.newaxis], y_train) # 训练模型，newaxis 将 1D 数组转为 2D 列向量
print("输出参数w：",model.coef_) # 输出模型参数w（权重/斜率），应接近 1.5
print("输出参数b：",model.intercept_) # 输出参数b（偏置/截距），应接近 0.2

# --- 3. 可视化预测结果 ---
X_test = np.linspace(0, 1, 100) # 生成测试点用于绘制曲线
plt.plot(X_test, model.predict(X_test[:, np.newaxis]), label="Model") # 模型预测曲线
plt.plot(X_test, true_fun(X_test), label="True function") # 真实函数曲线
plt.scatter(X_train,y_train) # 画出训练集的点
plt.legend(loc="best")
plt.show()


