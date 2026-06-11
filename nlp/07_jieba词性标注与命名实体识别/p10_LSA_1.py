"""
潜在语义分析（LSA）示例 —— 基于 SVD 分解。

教学重点：
1. 对词-文档矩阵（Term-Document Matrix）进行 SVD 分解
2. 截断奇异值保留前 k 个主题
3. 通过 L1/L2 范数提取每个主题的关键词
"""

import numpy as np

# --- 1. 构建词-文档矩阵 ---
# 此处使用一个示例的词-文档矩阵（Term-Document Matrix）
X = np.array([
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 1],
    [1, 1, 0, 0, 0],
    [0, 0, 1, 1, 0],
    [1, 0, 0, 1, 1],
    [0, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1],
], dtype=float)

# 提取的主题数
k = 2 

# --- 2. SVD 分解 ---
Up, Sp, Vp = np.linalg.svd(X, full_matrices=True)
print(f'左奇异矩阵 Up = \n{Up}')
print(f'奇异值向量 Sp = \n{Sp}')
print(f'右奇异矩阵 Vp = \n{Vp}')

# --- 3. 截断为前 k 个主题 ---
Sp = Sp[:k]
Up = Up[:, :k]

# --- 4. L1 范数归一化 ---
# 计算L1范数
UpAbs = np.abs(Up)
M = UpAbs / np.maximum(1e-6, np.sum(UpAbs, axis=1, keepdims=True))
print(f' L1范数 M = \n{M}')

# 根据L1范数计算每个主题的s=3个关键字
s = 3
value = np.sort(M, axis=0)[::-1][:s]
print(f'value=\n{value}')

# --- 5. L2 范数归一化（对比 L1 和 L2 的差异） ---
M = UpAbs / np.maximum(1e-6, np.sqrt(np.sum(UpAbs**2, axis=1, keepdims=True)))
print(f' L2范数 M = \n{M}')
