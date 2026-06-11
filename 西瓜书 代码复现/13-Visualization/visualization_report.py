#!/usr/bin/env python
# coding: utf-8

"""
机器学习可视化报告 —— 西瓜书代码复现

使用 Scikit-Plot 库对多种机器学习模型进行可视化评估。
教学重点：
    1. 学习曲线（Learning Curve）：观察训练集大小对模型性能的影响
    2. 特征重要性（Feature Importance）：了解哪些特征对预测最关键
    3. 混淆矩阵（Confusion Matrix）：直观展示分类错误的分布
    4. ROC / PR 曲线：评估分类器在不同阈值下的性能
    5. 轮廓分析（Silhouette Analysis）：评估聚类质量
    6. 可靠性曲线（Calibration Curve）：检验模型预测概率的可靠性
    7. KS 统计量、累积收益曲线、Lift 曲线：评估二分类模型的区分能力
    8. 手肘法（Elbow Method）：选择聚类的最优 K 值
    9. PCA 方差分析与二维投影可视化
Scikit-Plot 的地址：https://github.com/reiinakano/scikit-plot
Scikit-Plot 的官方文档：https://scikit-plot.readthedocs.io/en/stable/
"""

# ## 1 简介
#
# 本次主要通过使用```Scikit-Plot```的模块来介绍机器学习的相关可视化，```Scikit-Plot```主要包括以下几个部分：
# * estimators：用于绘制各种算法
# * metrics：用于绘制机器学习的onfusion matrix, ROC AUC curves, precision-recall curves等曲线
# * cluster：主要用于绘制聚类
# * decomposition：主要用于绘制PCA降维
#
# ```Scikit-Plot```的地址：https://github.com/reiinakano/scikit-plot
#
# ```Scikit-Plot```的官方文档：https://scikit-plot.readthedocs.io/en/stable/

# In[4]:


# --- 加载需要用到的模块 ---
import scikitplot as skplt # Scikit-Plot：机器学习可视化库

import sklearn
from sklearn.datasets import load_digits, load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split

# 集成学习模型
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, ExtraTreesClassifier
# 线性模型
from sklearn.linear_model import LinearRegression, LogisticRegression
# 聚类和降维
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import sys

print("Scikit Plot Version : ", skplt.__version__)
print("Scikit Learn Version : ", sklearn.__version__)
print("Python Version : ", sys.version)


# ## 2 加载数据集

# ### 2.1 手写数据集

# In[5]:


# --- 加载手写数字数据集：1797 张 8x8 图片，10 个类别（数字 0-9） ---
digits = load_digits()
X_digits, Y_digits = digits.data, digits.target

print("Digits Dataset Size : ", X_digits.shape, Y_digits.shape)

# stratify=Y_digits：按类别分层抽样，保证训练/测试集中各类别比例一致
X_digits_train, X_digits_test, Y_digits_train, Y_digits_test = train_test_split(X_digits, Y_digits,
                                                                                train_size=0.8,
                                                                                stratify=Y_digits,
                                                                                random_state=1)

print("Digits Train/Test Sizes : ",X_digits_train.shape, X_digits_test.shape, Y_digits_train.shape, Y_digits_test.shape)



# ### 2.2 肿瘤数据集

# In[6]:


# --- 加载乳腺癌数据集：569 个样本，30 个特征，2 个类别（良性/恶性） ---
cancer = load_breast_cancer()
X_cancer, Y_cancer = cancer.data, cancer.target

print("Feautre Names : ", cancer.feature_names)
print("Cancer Dataset Size : ", X_cancer.shape, Y_cancer.shape)
X_cancer_train, X_cancer_test, Y_cancer_train, Y_cancer_test = train_test_split(X_cancer, Y_cancer,
                                                                                train_size=0.8,
                                                                                stratify=Y_cancer,
                                                                                random_state=1)

print("Cancer Train/Test Sizes : ",X_cancer_train.shape, X_cancer_test.shape, Y_cancer_train.shape, Y_cancer_test.shape)


# ### 2.3 波斯顿房价数据集

# In[7]:


boston = load_boston()
X_boston, Y_boston = boston.data, boston.target

print("Boston Dataset Size : ", X_boston.shape, Y_boston.shape)

print("Boston Dataset Features : ", boston.feature_names)
X_boston_train, X_boston_test, Y_boston_train, Y_boston_test = train_test_split(X_boston, Y_boston,
                                                                                train_size=0.8,
                                                                                random_state=1)

print("Boston Train/Test Sizes : ",X_boston_train.shape, X_boston_test.shape, Y_boston_train.shape, Y_boston_test.shape)


# ## 3 性能可视化

# ### 3.1 学习曲线（Learning Curve）
# 学习曲线展示随着训练样本数量增加，模型在训练集和验证集上的表现变化。
# 可用于判断模型是否过拟合或欠拟合。

# In[14]:


# --- 分类学习曲线：Logistic 回归在手写数字数据集上 ---
# cv=7：7 折交叉验证
skplt.estimators.plot_learning_curve(LogisticRegression(), X_digits, Y_digits,
                                     cv=7, shuffle=True, scoring="accuracy",
                                     n_jobs=-1, figsize=(6,4), title_fontsize="large", text_fontsize="large",
                                     title="Digits Classification Learning Curve")
plt.show()

# --- 回归学习曲线：线性回归在波士顿房价数据集上 ---
skplt.estimators.plot_learning_curve(LinearRegression(), X_boston, Y_boston,
                                     cv=7, shuffle=True, scoring="r2", n_jobs=-1,
                                     figsize=(6,4), title_fontsize="large", text_fontsize="large",
                                     title="Boston Regression Learning Curve ");
plt.show()                                


# ### 3.2 特征重要性（Feature Importance）
# 特征重要性展示每个特征对模型预测的贡献程度，有助于理解模型和特征选择。

# In[18]:


# --- 随机森林回归器：在波士顿房价数据上评估特征重要性 ---
rf_reg = RandomForestRegressor()
rf_reg.fit(X_boston_train, Y_boston_train)
print(rf_reg.score(X_boston_test, Y_boston_test))
# --- 梯度提升分类器：在乳腺癌数据上评估特征重要性 ---
gb_classif = GradientBoostingClassifier()
gb_classif.fit(X_cancer_train, Y_cancer_train)
print(gb_classif.score(X_cancer_test, Y_cancer_test))

# 绘制两个模型的特征重要性对比图
fig = plt.figure(figsize=(15,6))

ax1 = fig.add_subplot(121)
skplt.estimators.plot_feature_importances(rf_reg, feature_names=boston.feature_names, # 波士顿房价特征
                                         title="Random Forest Regressor Feature Importance",
                                         x_tick_rotation=90, order="ascending", # 升序排列
                                         ax=ax1);

ax2 = fig.add_subplot(122)
skplt.estimators.plot_feature_importances(gb_classif, feature_names=cancer.feature_names, # 乳腺癌特征
                                         title="Gradient Boosting Classifier Feature Importance",
                                         x_tick_rotation=90,
                                         ax=ax2);

plt.tight_layout()
plt.show()


# ## 4 机器学习度量(metrics)

# ### 4.1 混淆矩阵（Confusion Matrix）
# 混淆矩阵展示模型预测结果与真实标签的对应关系。
# 对角线上的值为正确分类的数量，非对角线为误分类。

# In[19]:


# --- 训练 Logistic 回归并进行预测 ---
log_reg = LogisticRegression()
log_reg.fit(X_digits_train, Y_digits_train)
log_reg.score(X_digits_test, Y_digits_test)
Y_test_pred = log_reg.predict(X_digits_test)

# --- 绘制混淆矩阵：绝对值版和归一化版 ---
fig = plt.figure(figsize=(15,6))

ax1 = fig.add_subplot(121)
skplt.metrics.plot_confusion_matrix(Y_digits_test, Y_test_pred, # 绝对数量混淆矩阵
                                    title="Confusion Matrix",
                                    cmap="Oranges",
                                    ax=ax1)

ax2 = fig.add_subplot(122)
skplt.metrics.plot_confusion_matrix(Y_digits_test, Y_test_pred, # 归一化混淆矩阵（比例）
                                    normalize=True,
                                    title="Confusion Matrix",
                                    cmap="Purples",
                                    ax=ax2);
plt.show()


# ### 4.2 ROC 曲线与 AUC
# ROC 曲线展示不同分类阈值下 TPR（真正率）与 FPR（假正率）的关系。
# AUC（曲线下面积）越接近 1 表示模型性能越好。

# In[21]:


# 获取预测概率（每条数据属于每个类别的概率）
Y_test_probs = log_reg.predict_proba(X_digits_test)

skplt.metrics.plot_roc_curve(Y_digits_test, Y_test_probs, # 多分类 ROC 曲线（每个类别一条）
                       title="Digits ROC Curve", figsize=(12,6))
plt.show()


# ### 4.3 PR 曲线（Precision-Recall Curve）
# PR 曲线展示查准率（Precision）和查全率（Recall）之间的权衡关系。
# 适用于类别不平衡场景。

# In[23]:


skplt.metrics.plot_precision_recall_curve(Y_digits_test, Y_test_probs,
                       title="Digits Precision-Recall Curve", figsize=(12,6))
plt.show()


# ### 4.4 轮廓分析（Silhouette Analysis）
# 轮廓系数衡量样本与其所在簇的紧密度和与其他簇的分离度。
# 值越接近 1 表示聚类效果越好。

# In[24]:


kmeans = KMeans(n_clusters=10, random_state=1) # 将手写数字分为 10 个簇
kmeans.fit(X_digits_train, Y_digits_train)
cluster_labels = kmeans.predict(X_digits_test)
skplt.metrics.plot_silhouette(X_digits_test, cluster_labels,
                              figsize=(8,6))
plt.show()


# ### 4.5 可靠性曲线（Calibration Curve，Reliability Curves）
# 可靠性曲线检验模型预测概率是否准确反映真实概率。
# 理想情况下，曲线应接近对角线（完美校准）。

# In[25]:


# 训练 4 个不同模型，比较它们的预测概率校准程度
lr_probas = LogisticRegression().fit(X_cancer_train, Y_cancer_train).predict_proba(X_cancer_test)
rf_probas = RandomForestClassifier().fit(X_cancer_train, Y_cancer_train).predict_proba(X_cancer_test)
gb_probas = GradientBoostingClassifier().fit(X_cancer_train, Y_cancer_train).predict_proba(X_cancer_test)
et_scores = ExtraTreesClassifier().fit(X_cancer_train, Y_cancer_train).predict_proba(X_cancer_test)

probas_list = [lr_probas, rf_probas, gb_probas, et_scores]
clf_names = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Extra Trees Classifier']
skplt.metrics.plot_calibration_curve(Y_cancer_test,
                                     probas_list,
                                     clf_names, n_bins=15, # 将预测概率分为 15 个区间
                                     figsize=(12,6)
                                     )
plt.show()


# ### 4.6 KS 检验（Kolmogorov-Smirnov）
# KS 统计量衡量模型对正负样本的区分能力。
# KS 值 = max(TPR - FPR)，越大说明区分能力越强。

# In[26]:


rf = RandomForestClassifier()
rf.fit(X_cancer_train, Y_cancer_train)
Y_cancer_probas = rf.predict_proba(X_cancer_test)

skplt.metrics.plot_ks_statistic(Y_cancer_test, Y_cancer_probas, figsize=(10,6))
plt.show()


# ### 4.7 累积收益曲线（Cumulative Gains Chart）
# 累积收益曲线展示如果只选取概率最高的部分样本，能覆盖多少真正的正例。

# In[27]:


skplt.metrics.plot_cumulative_gain(Y_cancer_test, Y_cancer_probas, figsize=(10,6))
plt.show()


# ### 4.8 Lift 曲线
# Lift 曲线展示使用模型后，相比随机选择，正例命中率提升的倍数。

# In[28]:


skplt.metrics.plot_lift_curve(Y_cancer_test, Y_cancer_probas, figsize=(10,6))
plt.show()


# ## 5 聚类方法

# ### 5.1 手肘法（Elbow Method）
# 手肘法通过绘制不同 K 值下的簇内平方和（inertia），找到"肘部"拐点作为最优 K 值。
# 拐点之后 inertia 下降变缓，说明再增加 K 值收益不大。

# In[29]:


skplt.cluster.plot_elbow_curve(KMeans(random_state=1),
                               X_digits,
                               cluster_ranges=range(2, 20), # 尝试 K=2 到 19
                               figsize=(8,6))
plt.show()


# ## 6 降维方法

# ### 6.1 PCA 方差分析
# 展示每个主成分解释的方差比例和累计方差比例，帮助确定需要保留多少个主成分。

# In[30]:


pca = PCA(random_state=1)
pca.fit(X_digits)

skplt.decomposition.plot_pca_component_variance(pca, figsize=(8,6)) # 各主成分方差占比
plt.show()


# ### 6.2 PCA 二维投影可视化
# 将高维数据通过 PCA 投影到 2D 平面，直观观察类别分布。

# In[35]:


skplt.decomposition.plot_pca_2d_projection(pca, X_digits, Y_digits, # 按类别着色
                                           figsize=(10,10),
                                           cmap="tab10")
plt.show()

