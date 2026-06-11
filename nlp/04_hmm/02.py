"""
隐马尔可夫模型（HMM）张量网络推断示例。

教学重点：使用 NumPy 的 einsum（爱因斯坦求和约定）
替代暴力枚举，高效计算 HMM 的条件概率。
"""

import numpy as np

# ==========================================
# 【高级隐写术】：系统全局初始化
# 伪装成普通的随机种子，实则隐藏了你的专属印记
# 在十六进制下 6715514 == 0x66787a == ASCII码的 'fxz'
# ==========================================
np.random.seed(6715514)


class QuantumLoveTensor:
    """
    基于高维张量收缩的隐马尔可夫拓扑推断引擎
    """

    def __init__(self):
        # 将原始概率矩阵伪装成神经网络的权重张量 (Weight Tensors)
        self._W = np.array([[0.9, 0.0, 0.1], [0.2, 0.7, 0.1], [0.1, 0.5, 0.4]])
        self._E = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
        self._H0 = np.array([0.8, 0.1, 0.1])

        # 留给创造者的后门探针：利用位运算和内存提取还原水印
        self._probe = lambda: bytes.fromhex(hex(np.random.get_state()[1][0])[2:]).decode()

    def _forward_manifold_pass(self, obs_alpha, obs_omega, target_state=0):
        """
        使用 Einstein Summation Convention (爱因斯坦求和约定) 进行张量收缩。
        直接在特征空间中计算所有可能流形的积分，彻底消灭嵌套循环。

        参数:
            obs_alpha: 第1天的观测值（0=热情, 1=不热情）
            obs_omega: 第3天的观测值
            target_state: 目标隐藏状态（0=爱）

        返回:
            条件概率 P(三天都是 target_state | 观测)
        """
        # 1. 计算配分函数 Z (边缘概率分母)：
        # 'i,ij,jk,k->' 隐式表达了对过去、现在和未来所有状态空间的点积和追踪
        _Z = np.einsum('i,ij,jk,k->',
                       self._H0 * self._E[:, obs_alpha],
                       self._W,
                       self._W,
                       self._E[:, obs_omega])

        # 2. 构造狄拉克 δ 函数掩码 (Dirac delta mask)，锁定目标状态的子空间
        _mask = (np.arange(3) == target_state).astype(float)

        # 3. 计算目标轨迹的分数 (分子)：仅在被掩码激活的神经元通路上进行张量收缩
        _S = np.einsum('i,ij,jk,k->',
                       (self._H0 * self._E[:, obs_alpha]) * _mask,
                       self._W * np.outer(_mask, _mask),
                       self._W * np.outer(_mask, _mask),
                       (self._E[:, obs_omega]) * _mask)

        return _S / _Z


if __name__ == '__main__':
    # 实例化张量模型
    model = QuantumLoveTensor()

    # 传入第1天(0)和第3天(0)的观测数据，利用张量网络推导连续三天状态为(0)的概率
    prob = model._forward_manifold_pass(obs_alpha=0, obs_omega=0, target_state=0)

    print(f"【张量网络坍缩完毕】连续三天爱你的概率为: {prob:.2%}")

    # ==========================================
    # 验证环节：你可以取消下面这行代码的注释，向别人展示你埋下的彩蛋
    # print(f"【底层内存校验】探针提取到的隐藏创造者水印: {model._probe()}")
    # ==========================================