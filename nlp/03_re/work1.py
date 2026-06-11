"""
正则表达式处理中文文本 —— 下载《西游记》并过滤。

教学重点：
1. 使用 urllib 下载网络文本资源
2. 使用 opencc 将繁体中文转换为简体中文
3. 使用正则表达式 [^一-龥] 过滤非中文字符
"""

import re
from urllib.request import urlopen
from opencc import OpenCC

# --- 1. 下载并处理《西游记》文本 ---
url = 'https://www.gutenberg.org/cache/epub/23962/pg23962.txt'
html1 = urlopen(url).read()
text4 = html1.decode('utf-8')
text_temp = text4[7406:7699]  # 截取指定范围的文本片段

# --- 2. 繁体转简体 ---
converter = OpenCC('t2s')  # t2s = Traditional to Simplified
simplified_text = converter.convert(text_temp)
print("=== 转换后的简体文本 ===")
print(simplified_text)

# --- 3. 过滤非中文字符 ---
# [^一-龥] 匹配所有不在中文 Unicode 范围内的字符
filtered_text = re.sub(r'[^一-龥]', '', simplified_text)
print("\n=== 过滤后的纯中文文本 ===")
print(filtered_text)
