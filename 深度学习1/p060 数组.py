"""
NumPy 和 PyTorch 数组/张量入门。

教学重点：数组形状、索引和基础计算。
NumPy 和 PyTorch 的索引语法基本相同，都用 [行, 列] 来取值。
"""

import numpy as np
import torch

# --- 1. 标量和二维数组的创建 ---
print(np.array(123.45))               # NumPy 标量
print(torch.tensor(123.45))           # PyTorch 标量

np.array([[1.2,3.45],[6.78,9.]])      # NumPy 二维数组（未赋值给变量）
print(torch.tensor([[1.2,3.45],[6.78,9.]]))  # PyTorch 二维张量

# --- 2. 随机数组和基本索引 ---
a = np.random.rand(4,5)    # 4 行 5 列的随机数组
print(a)
b = np.random.rand(4,5)
print(b)

# 单个元素索引：a[行号, 列号]，从 0 开始计数
print(a[3,2])   # 第 4 行第 3 列
print(b[2,4])   # 第 3 行第 5 列

# --- 3. 切片索引：a[起始行:终止行, 起始列:终止列] ---
print(a[2:4,1:5])     # 第 3~4 行，第 2~5 列
print(b[2:3,1:3])     # 第 3 行，第 2~3 列
print(a[2:,3:5])      # 第 3 行起，第 4~5 列
print(b[2:,:3])       # 第 3 行起，第 1~3 列
print(a[:,2:4])       # 所有行，第 3~4 列
print(b[2:4,3:])      # 第 3~4 行，第 4 列起
print(a[3,:])         # 第 4 行所有列
print(b[:,:])         # 所有行所有列（完整复制）
print(a[2])           # 省略列号，取第 3 行整行

# --- 4. 赋值修改数组 ---
a[2,3] = 12.34567             # 修改单个元素
b[2,:] = 12.34567             # 修改整行
a[1::2,0:5:2] = 3.456         # 步长切片：奇数行、奇数列设为 3.456
b[:-1,:-2] = 78.9             # 除最后一行、除最后两列外的区域

# --- 5. 矩阵乘法 ---
a1 = np.random.uniform(-10,10,[2,3])   # 2x3 矩阵
a2 = np.random.uniform(-10,10,[3,5])   # 3x5 矩阵
a3 = np.matmul(a1,a2)                  # 矩阵乘法结果为 2x5
print("a1=\n",a1)
print("a1.shape=",a1.shape)
print("a2=\n",a2)
print("a2.shape=",a2.shape)
print("a3=\n",a3)
print("a3.shape=",a3.shape)

# --- 6. PyTorch Parameter（可训练参数） ---
init = torch.tenor([2.,3.,4.],[1.,-1.,-4.])
print(torch.nn.Parameter(init))
