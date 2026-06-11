#coding=utf-8
#Author:haobo
#Date:2022-4-23

"""
隐马尔可夫模型（HMM）示例 —— 西瓜书代码复现

使用 hmmlearn 库演示 HMM 的样本生成、参数学习和状态预测。
教学重点：
    1. HMM 由初始状态概率、状态转移概率矩阵和发射概率三部分组成
    2. 通过 EM 算法（Baum-Welch）学习模型参数
    3. 通过 Viterbi 算法预测最可能的隐状态序列
算法原理：
    HMM 的三要素：
    1) 初始状态概率 pi：pi_i = P(z_1 = i)，系统在时刻 1 处于状态 i 的概率
    2) 状态转移矩阵 A：a_ij = P(z_t = j | z_{t-1} = i)，从状态 i 转移到状态 j 的概率
    3) 发射概率（观测概率）B：b_j(x) = P(x | z = j)，在状态 j 下观测到 x 的概率
    高斯 HMM 假设发射概率为多元高斯分布。
"""
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm

# --- 1. 设置 HMM 参数（4 个隐状态，2 维观测） ---
# Prepare parameters for a 4-components HMM
# Initial population probability
startprob = np.array([0.6, 0.3, 0.1, 0.0]) # 初始状态概率：最可能从状态 0 开始
# The transition matrix, note that there are no transitions possible
# between component 1 and 3
transmat = np.array([[0.7, 0.2, 0.0, 0.1],   # 状态 0 → 主要保持在状态 0
                     [0.3, 0.5, 0.2, 0.0],   # 状态 1
                     [0.0, 0.3, 0.5, 0.2],   # 状态 2
                     [0.2, 0.0, 0.2, 0.6]])  # 状态 3
# The means of each component
means = np.array([[0.0, 0.0],    # 状态 0 的观测均值
                  [0.0, 11.0],   # 状态 1 的观测均值
                  [9.0, 10.0],   # 状态 2 的观测均值
                  [11.0, -1.0]]) # 状态 3 的观测均值
# The covariance of each component
covars = .5 * np.tile(np.identity(2), (4, 1, 1)) # 每个状态的协方差矩阵为 0.5*I

# --- 2. 构建生成模型并生成观测样本 ---
# Build an HMM instance and set parameters
gen_model = hmm.GaussianHMM(n_components=4, covariance_type="full")

# Instead of fitting it from the data, we directly set the estimated
# parameters, the means and covariance of the components
gen_model.startprob_ = startprob
gen_model.transmat_ = transmat
gen_model.means_ = means
gen_model.covars_ = covars

# Generate samples：根据模型参数生成 500 个观测点
X, Z = gen_model.sample(500) # X 为观测序列，Z 为对应的隐状态序列

# --- 3. 可视化生成的观测数据 ---
# Plot the sampled data
fig, ax = plt.subplots()
ax.plot(X[:, 0], X[:, 1], ".-", label="observations", ms=6,
        mfc="orange", alpha=0.7)

# Indicate the component numbers：标注各状态的均值位置
for i, m in enumerate(means):
    ax.text(m[0], m[1], 'Component %i' % (i + 1),
            size=17, horizontalalignment='center',
            bbox=dict(alpha=.7, facecolor='w'))
ax.legend(loc='best')
fig.show()


# --- 4. 使用 EM 算法学习模型参数 ---
# 尝试不同的隐状态数（3, 4, 5），通过交叉验证选择最优模型
scores = list()
models = list()
for n_components in (3, 4, 5):
    # define our hidden Markov model
    model = hmm.GaussianHMM(n_components=n_components,
                            covariance_type='full', n_iter=10) # n_iter：EM 迭代次数
    model.fit(X[:X.shape[0] // 2])  # 50/50 train/validate：前半训练，后半验证
    models.append(model)
    scores.append(model.score(X[X.shape[0] // 2:])) # 在验证集上计算对数似然
    print(f'Converged: {model.monitor_.converged}'
          f'\tScore: {scores[-1]}')

# get the best model：选择得分最高的模型
model = models[np.argmax(scores)]
n_states = model.n_components
print(f'The best model had a score of {max(scores)} and {n_states} '
      'states')

# --- 5. 使用 Viterbi 算法预测隐状态序列 ---
# use the Viterbi algorithm to predict the most likely sequence of states
# given the model
states = model.predict(X) # 维特比算法：找到最可能的隐状态路径



# --- 6. 对比真实状态与预测状态 ---
#让我们将我们的状态与生成的状态和我们的转换矩阵进行比较，来看我们的模型
# plot model states over time
fig, ax = plt.subplots()
ax.plot(Z, states) # 对比生成时的真实状态 Z 和恢复出的状态 states
ax.set_title('States compared to generated')
ax.set_xlabel('Generated State')
ax.set_ylabel('Recovered State')
fig.show()

# --- 7. 对比真实转移矩阵与学习到的转移矩阵 ---
# plot the transition matrix
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
ax1.imshow(gen_model.transmat_, aspect='auto', cmap='spring') # 真实转移矩阵
ax1.set_title('Generated Transition Matrix')
ax2.imshow(model.transmat_, aspect='auto', cmap='spring') # 学习到的转移矩阵
ax2.set_title('Recovered Transition Matrix')
for ax in (ax1, ax2):
    ax.set_xlabel('State To')
    ax.set_ylabel('State From')

fig.tight_layout()
fig.show()