"""
jieba 词性标注基础示例。

教学重点：使用 jieba.posseg 模块对中文文本进行分词和词性标注，
输出每个词语及其对应的词性标签（如 n=名词, v=动词 等）。
"""

import jieba.posseg as pseg

# --- 1. 待标注文本 ---
text = "美国SpaceX在最新一次Starship试飞中成功完成关键轨道测试，标志着人类迈向深空运输能力的重要一步。"

# --- 2. 分词并词性标注 ---
words = pseg.cut(text)

# --- 3. 输出每个词及其词性 ---
for word , flag in words:
    print(word,flag)
