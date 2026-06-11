"""
使用 jieba 分词库对中文文本进行分词和词性标注。

教学重点：
- jieba.load_userdict() 加载自定义词典提高分词准确性
- jieba.cut() 精确模式分词
- jieba.posseg.cut() 同时进行分词和词性标注
"""

# --- 1. 导入模块 ---
import jieba
import jieba.posseg as posseg

# --- 2. 加载自定义词典 ---
jieba.load_userdict("my_dict.txt")  # 加载用户自定义词典，提高专有名词识别率

# --- 3. 精确模式分词 ---
s = "刚结束中国之行没几天，布林背又将于当地时间29日马不停蹄地赶赴中东。"
words = jieba.cut(s)       # 默认精确模式
print(list(words))

# --- 4. 全模式分词（已注释） ---
#words = jieba.cut(s, cut_all=True)  # 全模式：扫描所有可能的词语
#print(list(words))

# --- 5. 词性标注 ---
ds = posseg.cut(s)  # 同时分词并标注词性（名词n、动词v、形容词a等）
for d in ds:
    print(d.word, "/", d.flag, end=" ")  # 输出：词语 / 词性
print()