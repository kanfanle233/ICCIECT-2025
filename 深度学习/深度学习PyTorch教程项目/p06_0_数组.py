"""
NumPy 与 PyTorch 数组（张量）操作进阶，涵盖矩阵乘法与 Parameter。
教学重点：二维数组创建、下标访问、切片、reshape、transpose、flatten、unsqueeze、repeat、矩阵乘法 matmul、nn.Parameter。
"""

import numpy as np
import torch

# --- 1. 基本数据类型：常数、一维数组、多维数组、随机数组 ---
# 常数
print(np.array(123.45))
print(torch.tensor(123.45))
# 一维数组
print(np.array([1.2, 3.45, 6.78]))
print(torch.tensor([1.2, 3.45, 6.78]))
# 多维数组
np.array([[1.2, 3.45], [6.78, 9.]])
print(torch.tensor([[1.2, 3.45], [6.78, 9.]]))
# 随机数组(由[0, 1)上的随机数构成的3行4列数组)
a = np.random.rand(4, 5)
print(a)
b = torch.rand(4, 5)
print(b)

# --- 2. 下标与切片 ---
# 下标  a[3][2]
print(a[3, 2])     # 从0开始数的第3行第2列数据
print(b[2, 4])     # 从0开始数的第2行第4列数据

# 数组切片
print(a[2:4, 1:5]) # 左上角在[2,1]的2行4列数据
print(b[2:3, 1:3]) # 左上角在[2,1]的1行2列数据
print(a[2:, 3:5])   # 左上角在[2,3]的2行2列数据
print(b[2:, :3])    # 左上角在[2,0]的2行3列数据
print(a[:, 2:4])    # 左上角在[0,2]的4行2列数据
print(b[2:4, 3:])   # 左上角在[2,3]的2行2列数据
print(a[:3, :])     # 左上角在[0,0]的3行5列数据
print(b[:, :])      # 左上角在[0,0]的4行5列数据
print(a[2])         # 等价于a[2, :]
print(b[2:4])       # 等价于b[2:4, :]
print(a[0, 0::2])
print(a[0, 1::2])


# --- 3. 维度、形状与整形 ---
# 维度(ndim)和形状(shape)
print(a.ndim)       # a的维度, 2
print(b.ndim)

print(a.shape)      # (4, 5)
print(b.shape[0])   # 4
print(b.shape[1])   # 5
print(len(b))       # 等价于b.shape[0]

# 整形(Reshape)
print(a.reshape([2, 10]))     # 形状从[4, 5]转为[2, 10]。既可以用列表
print(b.reshape((2, 2, 5)))    # 也可以用元组表示目标形状，整形前后要保证各维度长度的乘积相等.
                        # np和torch几乎所有操作都产生一个新数组，而不改变原数组
                        # 也就是说，a、b仍保持原形状[4, 5]。
print(a.reshape([2, -1, 5]))   # 等价于a.reshape([2, 2, 5])，-1代表自动计算。a保持原形状。
print(b.reshape([5, -1]))      # 等价于b.reshape([5, 4])，b保持原形状。

# --- 4. 转置、flatten、unsqueeze、repeat ---
# 转置(transpose), 结果不同
print(np.random.rand(2,3,4).transpose([1, 2, 0]))         # 第0/1/2个维度转位第2/0/1个维度
print(torch.rand(2, 3, 4).transpose(0, 2))     # 交换第0个和第2个维度

# 多维数组转为1维数组
print(a.flatten())    # 等价于a.reshape([-1])
print(b.flatten())

# 增加一个维度
print(np.expand_dims(a, 1)) # 在1号维度上增加一个长度为1的维度，形状从[4, 5]转为[4, 1, 5]
print(b.unsqueeze(2))            # 在2号维度上增加一个长度为1的维度，形状从[4, 5]转为[4, 5, 1]

# 指定维度重复若干倍, 参数含义不同
print(a.repeat(3, 1)) # 1号维度（及其数据）重复3倍，形状从[4, 5]转为[4,15]
print(b.repeat(1, 3))             # 0号和1号维度分别复制1、3倍，结果形状是[4, 15]

# 修改某行某列的值，PyTorch只有在张量是个常量时可以
a[2, 3] = 12.34567
b[2, :] = 12.34567         # 第2行所有数据改为12.34567。
a[1::2, 0:5:2] = 3.456     # 把第1、3行的第0、2、4个数改为3.456。
b[:-1, :-2] = 78.9         # 除最后一行外，每一行倒数第2个（从后往前应该从1开始数）数
                           # 之前的所有数改为78.9。

# --- 5. 矩阵乘法与 nn.Parameter ---
# 矩阵乘法
a1 = np.random.uniform(-10, 10, [2, 3])
a2 = np.random.uniform(-10, 10, [3, 5])
a3 = np.matmul(a1, a2)      # [2, 5]
print("a1=\n", a1)
print("a1.shape=", a1.shape)
print("a2=\n", a2)
print("a2.shape=", a2.shape)
print("a3=\n", a3)
print("a3.shape=", a3.shape)

# 参数，唯一PyTorch有而numpy没有的东西
init = torch.tensor([[2., 3., 4.], [1., -1., -4.]])
print(torch.nn.Parameter(init))  # 用指定初值创建一个参数
# 参数的本质是一个可以被优化的变量。所以它不同于Python的变量。