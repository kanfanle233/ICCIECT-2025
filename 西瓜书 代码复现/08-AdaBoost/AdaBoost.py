"""
AdaBoost 集成学习示例 —— 西瓜书代码复现

使用 scikit-learn 的 AdaBoostClassifier 在 Iris 数据集上进行分类。
教学重点：
    1. AdaBoost 通过加权组合多个弱学习器（如单层决策树）构建强学习器
    2. 每轮迭代增大被错误分类样本的权重，使下一轮更关注难分样本
    3. learning_rate 控制每个弱学习器的贡献权重
算法原理：
    AdaBoost 算法流程：
    1) 初始化样本权重 w_i = 1/N
    2) 对每一轮 t = 1, ..., T：
       a) 用当前权重训练弱学习器 h_t
       b) 计算加权错误率 e_t
       c) 计算弱学习器权重 a_t = 0.5 * ln((1-e_t)/e_t)
       d) 更新样本权重：正确分类的权重减小，错误分类的权重增大
    3) 最终分类器 H(x) = sign(sum(a_t * h_t(x)))
"""
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

# --- 1. 加载 Iris 数据集并划分 ---
iris = datasets.load_iris()
X = iris.data   # 150 个样本，4 个特征
y = iris.target # 3 个类别（0, 1, 2）

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("X_train:",len(X_train),"; X_test:",len(X_test),"; y_train:",len(y_train),"; y_test:",len(y_test))

# --- 2. 训练 AdaBoost 模型 ---
# Create adaboost object
# n_estimators=50：最多训练 50 个弱学习器
# learning_rate=1.5：学习率，控制每个弱学习器的贡献程度
Adbc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1.5)
# Train Adaboost
model = Adbc.fit(X_train, y_train)

# --- 3. 评估模型 ---
#Predict the response for test dataset
y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#('Accuracy:', 0.8888888888888888)