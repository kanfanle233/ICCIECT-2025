"""
基于 HMM（隐马尔可夫模型）的中文分词实现（四分类：BMES）。

教学重点：
1. 使用 BMES 标注体系训练 HMM 参数（初始概率、转移概率、发射概率）
2. 实现维特比（Viterbi）算法进行序列解码
3. 将训练参数保存为 JSON 文件以便复用
"""

import json
import math
import os

def train(corpus_path):
    """
    训练HMM模型，将初始状态概率分布、状态转移概率矩阵和发射概率矩阵写入JSON文件当中
    """
    pi_dict = {s: 0.0 for s in 'BMES'}
    A_dict = {s: {s_prime: 0.0 for s_prime in 'BMES'} for s in 'BMES'}
    B_dict = {s: {} for s in 'BMES'}
    state_count = {s: 0.0 for s in 'BMES'}
    line_count = 0

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            words = line.split()
            char_list = []
            state_list = []
            
            for word in words:
                char_list.extend(list(word))
                if len(word) == 1:
                    state_list.append('S')
                else:
                    state_list.extend(['B'] + ['M'] * (len(word) - 2) + ['E'])
            
            if len(char_list) == 0:
                continue
            
            pi_dict[state_list[0]] += 1
            line_count += 1
            
            for i in range(len(state_list)):
                s = state_list[i]
                c = char_list[i]
                state_count[s] += 1
                
                # 统计发射概率
                B_dict[s][c] = B_dict[s].get(c, 0) + 1
                
                # 统计转移概率
                if i > 0:
                    prev_s = state_list[i - 1]
                    A_dict[prev_s][s] += 1

    # 归一化为概率
    pi_prob = {s: (pi_dict[s] / line_count) if line_count > 0 else 0 for s in 'BMES'}
    A_prob = {s: {s_prime: (A_dict[s][s_prime] / state_count[s]) if state_count[s] > 0 else 0 
                  for s_prime in 'BMES'} for s in 'BMES'}
    B_prob = {s: {c: count / state_count[s] for c, count in B_dict[s].items()} for s in 'BMES'}
    
    # 写入JSON文件
    with open('pi.json', 'w', encoding='utf-8') as f:
        json.dump(pi_prob, f, ensure_ascii=False, indent=4)
    with open('A.json', 'w', encoding='utf-8') as f:
        json.dump(A_prob, f, ensure_ascii=False, indent=4)
    with open('B.json', 'w', encoding='utf-8') as f:
        json.dump(B_prob, f, ensure_ascii=False, indent=4)
    
    return pi_prob, A_prob, B_prob

def load_model():
    """读取训练好的JSON文件模型"""
    with open('pi.json', 'r', encoding='utf-8') as f:
        pi_prob = json.load(f)
    with open('A.json', 'r', encoding='utf-8') as f:
        A_prob = json.load(f)
    with open('B.json', 'r', encoding='utf-8') as f:
        B_prob = json.load(f)
    return pi_prob, A_prob, B_prob

# 严格按照学习到的书本上的原版代码（无修改，完全复制图片1的代码逻辑）
def viterbi(text, states, start_prob, trans_prob, emit_prob):
    # 初始化
    V = [{}]
    path = {}
    # 初始时刻
    for state in states:
        V[0][state] = start_prob[state] * emit_prob[state].get(text[0], 0)
        path[state] = [state]
    # 动态规划
    for t in range(1, len(text)):
        V.append({})
        new_path = {}
        for state in states:
            prob, prev_state = max(
                [(V[t - 1][prev_state] * trans_prob[prev_state].get(state, 0) * emit_prob[state].get(text[t], 0), prev_state)
                 for prev_state in states])
            
            V[t][state] = prob
            new_path[state] = path[prev_state] + [state]
        path = new_path
    # 终止时刻
    prob, state = max((V[len(text) - 1][state], state) for state in states)
    seg_list = path[state]
    return seg_list

# 参考图片3提供的执行包裹接口风格
def get_result(text, pi, A, B):
    states = ['B', 'M', 'E', 'S']
    return viterbi(text, states, pi, A, B)

def cut(text, corpus_path):
    """
    满足最初要求(3)的需求接口，调用封装好的结果获取内容
    """
    if not os.path.exists('pi.json') or not os.path.exists('A.json') or not os.path.exists('B.json'):
        pi, A, B = train(corpus_path)
    else:
        pi, A, B = load_model()
        
    if not text:
        return []
        
    return get_result(text, pi, A, B)

if __name__ == '__main__':
    # path路径是子文件里面的trainCorpus
    corpus_path = '第4章/trainCorpus.txt'
    
    # 获取参数
    if not os.path.exists('pi.json') or not os.path.exists('A.json') or not os.path.exists('B.json'):
        pi, A, B = train(corpus_path)
    else:
        pi, A, B = load_model()
        
    # 简化输出，学习图3的格式打印输出，去除内部繁杂的循环打印
    print(f'pi=\n{pi}, \nA=\n{A}, \nB=\n{B}')
    
    text = "中文分词的主要任务是识别和切分出文本中的单个词语，这些词语是语言的基本语义单位。正确的分词结果对于后续的语言处理任务（如句法分析、语义理解、机器翻译等）而言是非常重要的。错误的分词结果可能导致误差在后续处理环节不断累积，严重影响整个NLP 系统的性能。"
    
    seg_list = get_result(text, pi, A, B)
    
    # 按照图片3最下方的遍历结果输出方式处理分词
    for i in range(len(seg_list)):
        if seg_list[i] in ('S', 'E'):
            print(text[i], end=' ')
        else:
            print(text[i], end='')
    
    # 输出换行保证控制台整洁
    print()
