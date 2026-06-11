"""
Bagging 与随机森林示例 —— 西瓜书代码复现

对比单棵决策树、Bagging 集成和随机森林在葡萄酒数据集上的分类性能。
教学重点：
    1. Bagging：通过有放回抽样生成多个训练子集，分别训练基学习器后投票/平均
    2. 随机森林：在 Bagging 基础上，每次划分节点时随机选择部分特征，进一步降低方差
    3. n_estimators（基学习器数量）对集成效果的影响
算法原理：
    Bagging 降低方差：通过多次采样训练不同模型，取平均减少波动。
    随机森林在 Bagging 基础上引入特征随机性，使各树之间差异更大，
    从而进一步降低过拟合风险，提升泛化能力。
"""
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# --- 1. 加载数据并训练基决策树 ---
wine = load_wine()#使用葡萄酒数据集
print(f"所有特征：{wine.feature_names}")
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
#构建并训练决策树分类器，这里特征选择标准使用基尼指数，树的最大深度为1（弱学习器）
base_model = DecisionTreeClassifier(max_depth=1, criterion='gini',random_state=1).fit(X_train, y_train)
y_pred = base_model.predict(X_test)#对测试集进行预测
print(f"决策树的准确率：{accuracy_score(y_test,y_pred):.3f}")

# --- 2. Bagging 集成学习 ---
from sklearn.ensemble import BaggingClassifier
# 建立Bagging分类器，每个基本分类模型为前面训练的决策树模型，基学习器个数为50
model = BaggingClassifier(base_estimator=base_model,
                            n_estimators=50,
                            random_state=1)
model.fit(X_train, y_train)# 训练：对 50 个基学习器分别在不同采样子集上训练
y_pred = model.predict(X_test)# 预测：50 个基学习器投票决定最终类别
print(f"BaggingClassifier的准确率：{accuracy_score(y_test,y_pred):.3f}")


# --- 3. 测试 Bagging 中基学习器个数对性能的影响 ---
x = list(range(2, 102, 2))  # 估计器个数即n_estimators，在这里我们取[2,102]的偶数
y = []

for i in x:
    model = BaggingClassifier(base_estimator=base_model,
                              n_estimators=i,

                              random_state=1)

    model.fit(X_train, y_train)
    model_test_sc = accuracy_score(y_test, model.predict(X_test))
    y.append(model_test_sc)

plt.style.use('ggplot')
plt.title("Effect of n_estimators", pad=20)
plt.xlabel("Number of base estimators")
plt.ylabel("Test accuracy of BaggingClassifier")
plt.plot(x, y)
plt.show() # 观察：随着基学习器数量增加，准确率先上升后趋于稳定

## --- 4. 随机森林 ---
# 随机森林 = Bagging + 特征随机选择（每次划分只考虑 sqrt(n_features) 个特征）

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
                            n_estimators=50,
                            random_state=1)
model.fit(X_train, y_train)# 训练
y_pred = model.predict(X_test)# 预测
print(f"RandomForestClassifier的准确率：{accuracy_score(y_test,y_pred):.3f}")

# --- 5. 测试随机森林中基学习器个数对性能的影响 ---
x = list(range(2, 102, 2))  # 估计器个数即n_estimators，在这里我们取[2,102]的偶数
y = []

for i in x:
    model = RandomForestClassifier(
        n_estimators=i,

        random_state=1)

    model.fit(X_train, y_train)
    model_test_sc = accuracy_score(y_test, model.predict(X_test))
    y.append(model_test_sc)

plt.style.use('ggplot')
plt.title("Effect of n_estimators", pad=20)
plt.xlabel("Number of base estimators")
plt.ylabel("Test accuracy of RandomForestClassifier")
plt.plot(x, y)
plt.show()