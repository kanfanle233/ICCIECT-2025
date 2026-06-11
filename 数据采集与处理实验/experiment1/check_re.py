"""
正则表达式基础语法练习。

教学重点：
- . 匹配任意字符
- * 零次或多次、+ 一次或多次、? 零次或一次
- {n} 精确次数、{n,m} 范围次数
- [] 字符集、^ 行首、$ 行尾
"""

# --- 1. 导入模块 ---
import re

# --- 2. 点号（.）匹配任意字符 ---
print(re.findall('m...e', 'cat and mouse and mouse'))  # 匹配 mouse（m后跟3个任意字符再跟e）
print(re.findall('m..e', 'cat and mouse'))              # 匹配 moue? 不匹配 mouse（需要恰好2个字符）

# --- 3. 量词：* + ? ---
print(re.findall('ca*t', 'caaaaaat and mouse and cat'))   # ca*t：a 出现零次或多次
print(re.findall('ca?t', 'cat and mouse and caaaaat'))     # ca?t：a 出现零次或一次，只匹配 cat 和 ct（隐含）
print(re.findall('ca?t', 'caat and mouse and caaaaat'))    # 不匹配 caat（a 出现了两次）

# --- 4. 更多量词示例 ---
print(re.findall('ca?t', 'cat and mouse and caaaaat'))
print()
print(re.findall('ca+t', 'ct and mouse and cat and caaaaat'))      # ca+t：a 至少出现一次
print(re.findall('ca{5}t', 'ct and mouse and cat and caaaaat'))     # ca{5}t：a 恰好出现 5 次
print(re.findall('ca{0,5}t', 'ct and mouse and cat and caaaaat'))   # ca{0,5}t：a 出现 0 到 5 次

# --- 5. 字符集 [] ---
print(re.findall('a[m,n]d', 'aad, abd, acd, add, amd, and, apd'))  # 匹配 a 后跟 m 或 n 再跟 d
print(re.findall('[a-z]d', 'aad, abd, acd, add, amd, and, apd'))    # 匹配任意小写字母后跟 d
print(re.findall('[c,a]{1,5}t', 'ct at and mouse and cat and caaaaat'))    # c 或 a 出现 1~5 次后跟 t

# --- 6. 锚点 ^ 和 $ ---
print(re.findall('[c,a]{1,5}t$', 'ct at and mouse and cat and caaaaat'))   # $ 匹配行尾
print(re.findall('^[c,a]{1,5}t', 'ct at and mouse and cat and caaaaat'))   # ^ 匹配行首