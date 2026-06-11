"""
唐诗 LSTM 样本处理脚本。

作用：
把文本里的每个汉字转换成整数编号，因为神经网络不能直接计算汉字，只能计算数字。

核心数据结构：
- ch_dict: {'汉字': [编号, 出现次数]}
- ch_order: ['第0号汉字', '第1号汉字', ...]
- samples: 每首诗对应的编号数组，例如 [0, 5, 9, 2]
"""

import numpy as np

sample_path = '全唐诗_7X4.txt'

def get_id(ch, ch_dict, ch_order):
    """返回汉字编号；如果是新汉字，就先加入字典。"""
    if ch in ch_dict:
        id_num = ch_dict[ch]
        id_num[1] += 1
        id = id_num[0]
    else:
        id = len(ch_dict)
        ch_dict[ch] = [id, 1]
        ch_order.append(ch)
    return id

def get_samples():
    """读取诗歌文本，并把每首诗转换成整数数组。"""
    ch_dict = dict()
    ch_order = []
    samples = []

    # ✅ 显式指定 UTF-8 编码
    with open(sample_path, encoding='utf-8') as fp:
        poems = fp.readlines()

    for p in poems:
        p = p.rstrip()
        print(p)
        poem = [get_id(ch, ch_dict, ch_order) for ch in p]
        samples.append(np.array(poem))

    print(f'{len(poems)} poems')
    return samples, ch_dict, ch_order


if __name__ == '__main__':
    samples, ch_dict, ch_order = get_samples()
    print(f'{len(ch_order)} 汉字')
    print(ch_dict['山'])
    print(ch_order[ch_dict['山'][0]])
