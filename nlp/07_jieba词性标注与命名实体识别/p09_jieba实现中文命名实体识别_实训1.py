"""
使用 jieba 实现中文命名实体识别（NER）。

教学重点：利用 jieba.posseg 的词性标注结果，
筛选出人名(nr)、地名(ns)、机构名(nt)、专名(nz)等命名实体。
"""

import jieba.posseg as pseg

# 待识别语句
text = "大家一致表示，要深刻领悟“两个确立”的决定性意义，切实增强“四个意识”、坚定“四个自信”、做到“两个维护”，始终牢记空谈误国、实干兴邦，蹲厉奋发、勇毅前行，用工作体现忠诚老实、用发展体现担当负责、用解决问题体现落实成效，以广东富有创造力的实践，推动报告描绘的宏伟蓝图变成美好现实，以广东创造的新辉煌，在我国现代化建设恢宏画卷上写下增光添彩的一笔。"

# 定义命名实体的词性标识（通常包括：nr人名，ns地名，nt机构团体名，nz其他专名）
ne_tags = {'nr', 'ns', 'nt', 'nz'}

# 使用jieba.posseg进行词性标注
words = pseg.cut(text)

# 提取并打印命名实体
print("中文命名实体识别结果：")
for word, flag in words:
    # 判断词性是否属于命名实体
    if flag in ne_tags or flag.startswith('nr') or flag.startswith('ns') or flag.startswith('nt') or flag.startswith('nz'):
        print(f"实体: {word:<5} \t 词性: {flag}")
