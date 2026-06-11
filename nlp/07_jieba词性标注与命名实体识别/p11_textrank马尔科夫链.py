"""
TextRank 算法核心 —— 马尔科夫链迭代求稳态分布。

教学重点：
1. 构建转移概率矩阵（行归一化）
2. 使用阻尼因子 d 的 PageRank 迭代公式
3. 通过多次迭代收敛到稳态概率分布
"""

import  numpy as np

# --- 1. 构建邻接矩阵（表示节点间的连接关系） ---
A = [
    [0,1,0,1],
    [3,0,5,2],
    [3,0,0,2],
    [0,0,0,0],
]
# --- 2. 行归一化为转移概率矩阵 ---
A = A / np.maximum(1e-5,np.sum(A,axis=1,keepdims=True))
print(f'A: \n{A}')

# --- 3. PageRank 迭代 ---
pi = [1,1,1,1]  # 初始概率分布（均匀分布）
d = 0.85         # 阻尼因子（通常取 0.85）

# 迭代 100 次直至收敛
for i in range(100):
    # PageRank 公式：pi_new = d * pi * A + (1-d) * 均匀分布
    pi2 = np.matmul(pi,A)*d + (1-d)
    print(f'{i+1}.pi: \n{pi2}')
    pi = pi2