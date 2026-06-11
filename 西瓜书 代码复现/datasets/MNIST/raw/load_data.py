#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-01-30 09:31:34
LastEditor: JiangJi
LastEditTime: 2023-01-30 09:31:35
Discription:
'''
"""
MNIST 数据集加载工具 —— 西瓜书代码复现

提供两种加载 MNIST 数据集的方法：
    1. load_local_mnist：从本地 .gz 压缩文件读取（推荐，无需联网）
    2. load_online_data：通过 Keras 在线下载（需要网络和 Keras 库）
教学重点：
    1. MNIST 是经典的手写数字数据集，60000 训练 + 10000 测试
    2. 图像为 28x28 灰度图，展平后为 784 维向量
    3. 像素值归一化到 [0,1] 可加速模型收敛
    4. One-hot 编码将类别标签转为二进制向量
"""
import numpy as np
from struct import unpack
import gzip
import os


def __read_image(path):
    """读取 MNIST 图像 .gz 压缩文件，返回 (num_images, 784) 的 numpy 数组"""
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16)) # 解析文件头：魔数、图片数、28、28
        img = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28) # 将字节数据转为像素矩阵
    return img


def __read_label(path):
    """读取 MNIST 标签 .gz 压缩文件，返回标签数组（0-9）"""
    with gzip.open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8)) # 解析文件头：魔数、标签数
        lab = np.frombuffer(f.read(), dtype=np.uint8)
        # print(lab[1])
    return lab


def __normalize_image(image):
    """将图像像素值从 [0,255] 归一化到 [0,1]，加速模型收敛
    Args:
        image (np.ndarray): 形状为 (N, 784) 的图像矩阵
    Returns:
        np.ndarray: 归一化后的 float32 图像矩阵
    """
    img = image.astype(np.float32) / 255.0
    return img


def __one_hot_label(label):
    """将数字标签进行 one-hot 编码
    Args:
        label (np.ndarray): 输入为 0-9 的数字标签
    Returns:
        np.ndarray: one-hot 编码矩阵，如数字 2 → [0,0,1,0,0,0,0,0,0,0]
    """
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab


def load_local_mnist(x_train_path=os.path.dirname(__file__)+'/train-images-idx3-ubyte.gz', y_train_path=os.path.dirname(__file__)+'/train-labels-idx1-ubyte.gz', x_test_path=os.path.dirname(__file__)+'/t10k-images-idx3-ubyte.gz', y_test_path=os.path.dirname(__file__)+'/t10k-labels-idx1-ubyte.gz', normalize=True, one_hot=True):
    """从本地 .gz 文件加载 MNIST 数据集
    Args:
        x_train_path (str): 训练图像文件路径
        y_train_path (str): 训练标签文件路径
        x_test_path (str): 测试图像文件路径
        y_test_path (str): 测试标签文件路径
        normalize (bool): 是否将像素值归一化到 [0,1]，默认 True
        one_hot (bool): 是否将标签转为 one-hot 编码，默认 True
    Returns:
        tuple: ((训练图像, 训练标签), (测试图像, 测试标签))
               训练集 60000 样本，测试集 10000 样本，每行 784=28*28 维
    """
    image = {
        'train': __read_image(x_train_path),
        'test': __read_image(x_test_path)
    }

    label = {
        'train': __read_label(y_train_path),
        'test': __read_label(y_test_path)
    }

    if normalize:
        for key in ('train', 'test'):
            image[key] = __normalize_image(image[key])

    if one_hot:
        for key in ('train', 'test'):
            label[key] = __one_hot_label(label[key])

    return (image['train'], label['train']), (image['test'], label['test'])



def load_online_data():  # categorical_crossentropy
    """通过 Keras 在线下载 MNIST 数据集（需要网络连接）
    Returns:
        tuple: ((训练图像, 训练标签), (测试图像, 测试标签))
    """
    from keras.datasets import mnist
    from keras.utils import np_utils
    import numpy as np
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    number = 10000
    x_train, y_train = x_train[0:number], y_train[0:number]
    x_train = x_train.reshape(number, 28 * 28) # 展平为 784 维向量
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, 10) # one-hot 编码
    y_test = np_utils.to_categorical(y_test, 10)
    x_test = np.random.normal(x_test)  # 加噪声，模拟真实场景中的数据扰动

    x_train, x_test = x_train / 255, x_test / 255 # 归一化到 [0,1]

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = load_local_mnist()
    print(f"训练集形状: {x_train.shape}, 标签形状: {y_train.shape}")
    print(f"测试集形状: {x_test.shape}, 标签形状: {y_test.shape}")