"""
jieba 分词与词频统计实训。

教学重点：
1. 使用 jieba 对中文文本进行分词
2. 加载停用词表过滤无意义词语
3. 统计词频并提取 Top-N 高频词
"""

import jieba


def word_extract():
    # （1）读取 flightnews.txt 文件。
    corpus = []
    # 因为文件在同一个目录下，直接写文件名作为相对路径即可
    path = 'flightnews.txt'
    content = ''
    # 使用 utf-8 编码读取，并忽略可能出现的编码错误
    for line in open(path, 'r', encoding='utf-8', errors='ignore'):
        line = line.strip()
        content += line
    corpus.append(content)

    # （2）加载停用词 stopword.txt 文件，使用 jieba 对新闻文本进行分词。
    stop_words = []
    # 同理，停用词文件也在同一目录下
    path_stopword = 'stopword.txt'
    # 按照书本结构读取停用词表
    for line in open(path_stopword, 'r', encoding='utf-8', errors='ignore'):
        line = line.strip()
        stop_words.append(line)

    # 使用 jieba 进行分词并去停用词
    split_words = []
    word_list = jieba.cut(corpus[0])

    for word in word_list:
        # 如果分出的词不在停用词表中，且为了严谨稍微过滤掉纯空白字符（可选）
        if word not in stop_words and word.strip() != '':
            split_words.append(word)

    # （3）提取出现频次最高的前 10 个词语。
    dic = {}
    word_num = 10

    # 统计词频
    for word in split_words:
        # 字典的 get 方法：如果 word 不在 dic 中则返回 0，然后加 1；在则取出原值加 1
        dic[word] = dic.get(word, 0) + 1

    # 将字典转为列表并按词频（即 x[1]）降序排序，最后切片提取前 10 个
    freq_word = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:word_num]

    # 结果展示输出
    print('样本：' + corpus[0])
    print('样本分词效果：' + '/ '.join(split_words))
    print('样本前 10 个高频词：' + str(freq_word))


# 调用运行提取函数
if __name__ == '__main__':
    word_extract()