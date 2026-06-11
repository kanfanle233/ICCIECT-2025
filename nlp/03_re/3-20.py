"""
正则表达式匹配二进制数。

教学重点：使用分组捕获和量词匹配二进制数字串。
"""

import re

# --- 1. 准备测试数据 ---
nums = [0, 5, 11, 1024]
P = r"0b0*(0|1[01]*)"  # 匹配二进制数：前导零 + (0 或 1开头的01串)

# --- 2. 逐个测试匹配结果 ---
for n in nums:
    s = bin(n).replace("0b", "")   # 转为二进制字符串并去掉前缀
    m = re.match(P, s)

    print("n =", n)
    print("binary =", s)

    if m:
        print("match =", m.group(), "length =", len(m.group()))
    else:
        print("match = None")

    print()
