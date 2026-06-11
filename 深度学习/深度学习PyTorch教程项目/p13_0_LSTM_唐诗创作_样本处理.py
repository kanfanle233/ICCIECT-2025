"""
唐诗数据集预处理模块，将古诗文本转为数字序列。
教学重点：
1) 字符到整数 ID 的映射（词典构建）
2) 古诗文本按行读取并编码为 numpy 数组序列
3) 为 LSTM 语言模型准备训练样本
"""
import numpy as np

sample_path = '../资源/全唐诗_7X4.txt'

def get_id(ch, ch_dict, ch_order):
    """获取字符的整数 ID，同时更新字典和出现次数统计"""
    if ch in ch_dict:
        id_num = ch_dict[ch]
        id_num[1] += 1      # 增加该字的出现次数
        id = id_num[0]       # 取已有 ID
    else:
        id = len(ch_dict)    # 新字符：ID = 当前字典大小
        ch_dict[ch] = [id, 1]
        ch_order.append(ch)  # 按出现顺序记录字符
    return id

def get_samples():
    """读取唐诗文件，将每首诗编码为整数序列

    返回：
        samples: 每首诗的整数 ID 序列列表
        ch_dict: 字符 -> [ID, 出现次数] 的字典
        ch_order: 按首次出现顺序排列的字符列表
    """
    ch_dict = dict()
    ch_order = []
    samples = []
    with open(sample_path, encoding='utf-8') as fp:
        poems = fp.readlines()
    for p in poems:
        p = p.rstrip()  # 去除行尾换行符
        print(p)
        poem = [get_id(ch, ch_dict, ch_order) for ch in p]  # 将每首诗转为 ID 序列
        samples.append(np.array(poem))
    print(f'{len(poems)} poems')
    return samples, ch_dict, ch_order

if __name__ == '__main__':
    samples, ch_dict, ch_order = get_samples()
    print(f'{len(ch_order)}汉字')  # 打印词汇表大小
    print(ch_dict['山'])           # 查看"山"字的 ID 和出现次数
    print(ch_order[ch_dict['山'][0]])  # 验证：通过 ID 反查字符
