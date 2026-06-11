"""
决策树分类示例 —— 西瓜书代码复现

使用 scikit-learn 的 DecisionTreeClassifier 对 Iris（鸢尾花）数据集进行分类。
教学重点：
    1. 决策树通过递归选择最优特征进行划分，构建树状分类规则
    2. 使用 violinplot / pointplot / Andrews 曲线进行数据探索可视化
    3. 训练决策树并以文字和图形方式输出决策规则
算法原理：
    决策树在每个节点选择使划分后"纯度"最高的特征和阈值。
    常用的纯度指标有：信息增益（ID3）、增益率（C4.5）、基尼指数（CART）。
    本例使用 CART 算法（基尼指数），限制最大深度为 3。
"""
import seaborn as sns
from pandas import plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree

# --- 1. 加载 Iris 数据集并探索 ---
data = load_iris() # 加载数据集：150 个样本，4 个特征，3 个类别
# 转换成.DataFrame形式
df = pd.DataFrame(data.data, columns = data.feature_names)
# 添加品种列
df['Species'] = data.target
# 查看数据集信息
print(f"数据集信息：\n{df.info()}")
# 查看前5条数据
print(f"前5条数据：\n{df.head()}")
# 查看各特征列的摘要信息
df.describe()

# --- 2. 数据可视化探索 ---
# 设置颜色主题
antV = ['#1890FF', '#2FC25B', '#FACC14', '#223273', '#8543E0', '#13C2C2', '#3436c7', '#F04864']
# 绘制violinplot：小提琴图展示每个特征在不同类别下的分布
f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
sns.despine(left=True) # 删除上方和右方坐标轴上不需要的边框，这在matplotlib中是无法通过参数实现的
sns.violinplot(x='Species', y=df.columns[0], data=df, palette=antV, ax=axes[0, 0]) # 花萼长度
sns.violinplot(x='Species', y=df.columns[1], data=df, palette=antV, ax=axes[0, 1]) # 花萼宽度
sns.violinplot(x='Species', y=df.columns[2], data=df, palette=antV, ax=axes[1, 0]) # 花瓣长度
sns.violinplot(x='Species', y=df.columns[3], data=df, palette=antV, ax=axes[1, 1]) # 花瓣宽度
plt.show()
# 绘制pointplot：点图展示每个特征在不同类别下的均值和置信区间
f, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
sns.despine(left=True)
sns.pointplot(x='Species', y=df.columns[0], data=df, color=antV[1], ax=axes[0, 0])
sns.pointplot(x='Species', y=df.columns[1], data=df, color=antV[1], ax=axes[0, 1])
sns.pointplot(x='Species', y=df.columns[2], data=df, color=antV[1], ax=axes[1, 0])
sns.pointplot(x='Species', y=df.columns[3], data=df, color=antV[1], ax=axes[1, 1])
plt.show()
# g = sns.pairplot(data=df, palette=antV, hue= 'Species')
# 安德鲁曲线：将多维特征映射为一条曲线，用于观察类别间的可分性
plt.subplots(figsize = (8,6))
plotting.andrews_curves(df, 'Species', colormap='cool')

plt.show()


# --- 3. 训练决策树分类器 ---
# 加载数据集
data = load_iris()
# 转换成.DataFrame形式
df = pd.DataFrame(data.data, columns = data.feature_names)
# 添加品种列
df['Species'] = data.target

# 用数值替代品种名作为标签
target = np.unique(data.target)
target_names = np.unique(data.target_names)
targets = dict(zip(target, target_names))
df['Species'] = df['Species'].replace(targets)

# 提取数据和标签
X = df.drop(columns="Species") # 特征矩阵：花萼/花瓣的长度和宽度
y = df["Species"]              # 标签向量：鸢尾花种类
feature_names = X.columns
labels = y.unique()

# 划分训练集（60%）和测试集（40%）
X_train, test_x, y_train, test_lab = train_test_split(X,y,
                                                 test_size = 0.4,
                                                 random_state = 42)
# 构建决策树：max_depth=3 限制树深度防止过拟合
model = DecisionTreeClassifier(max_depth =3, random_state = 42)
model.fit(X_train, y_train)
# --- 4. 可视化决策树 ---
# 以文字形式输出树
text_representation = tree.export_text(model)
print(text_representation)
# 用图片画出决策树结构
plt.figure(figsize=(30,10), facecolor ='g') #
a = tree.plot_tree(model,
                   feature_names = feature_names, # 特征名称
                   class_names = labels,          # 类别名称
                   rounded = True,                # 圆角矩形
                   filled = True,                 # 按类别着色
                   fontsize=14)
plt.show()                                          