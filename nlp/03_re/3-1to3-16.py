"""
正则表达式基础操作示例（代码 3-1 至 3-16）。

教学重点：演示 Python re 模块的核心函数，
包括 match、search、findall、sub、finditer、split，
以及量词、字符集、锚点等正则表达式语法。
"""

# --- 代码 3-1：re.match 从字符串开头匹配 ---
#3-1
import re
text1 = ('自然语言处理是研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法'
         '自然语言处理是一门融语言学、计算机科学、数学于一体的科学。')
print('匹配的结果是：',re.match('自然语言处理',text1))
print('匹配的结果是：',re.match('语言处理',text1))
# --- 代码 3-2：按句号分割后逐句匹配 ---
#3-2
p_string=  text1.split('。')
for line in p_string:
    if re.match('自然语言处理',line) is not None:
        print(line)
# --- 代码 3-3：re.search 在整个字符串中搜索（不限于开头） ---
#3-3
print(re.search('通信',text1))
import re
#3-4
# 在字符串中搜索匹配的子串
result = re.search(r'(\d+)-(\d+)-(\d+)', '2023-10-30')
if result:
    # 获取整个匹配到的子串
    print(result.group())  # 输出: 2023-10-30
    # 获取 search 函数中第一个括号内的子串
    print(result.group(1))  # 输出: 2023
    # 获取 search 函数中第二个括号内的子串
    print(result.group(2))  # 输出: 10
    # 获取 search 函数中第三个括号内的子串
    print(result.group(3))  # 输出: 30

# --- 代码 3-5：re.findall 查找所有匹配的子串 ---
#3-5
print(re.findall('计算机',text1))

# --- 代码 3-6：re.sub 替换匹配的子串 ---
#3-6
print(re.sub('自然语言处理','NLP',text1))

#3-7
# 定义要匹配的正则表达式模式
pattern = r"自然语言处理"
# 使用finditer函数进行迭代搜索
matches = re.finditer(pattern, text1)
# 遍历匹配结果并输出每个匹配的起始位置和结束位置
for match in matches:
    print("匹配文本：", match.group())
    print("匹配起始位置：", match.start())
    print("匹配结束位置：", match.end())
    print("-------------------------")

# --- 代码 3-8：re.split 按正则表达式分割字符串 ---
#3-8
pattern = "[ ，。]"
result = re.split(pattern, text1)
print(result)

# --- 代码 3-9：量词演示（? * + {n} 以及与 . 的组合） ---
#3-9
import re
# 唐初著名诗人刘希夷的诗《代悲白头翁》中截取和拼接的两句
text2 = '今年花落颜色改，明年花开复谁在？年年岁岁花相似，岁岁年年人不同。'
re.findall('年?', text2)  # "年" 至多出现 1 次
print(re.findall('年*', text2))  # "年" 可以出现 0 次、1 次或多次
re.findall('年+', text2)  # "年" 可以出现 1 次或多次
re.findall('年{1}', text2)  # "年" 正好出现 1 次
re.findall('年{2}', text2)  # "年" 正好出现 2 次
re.findall('年{0,1}', text2)  # "年" 出现 0 次或 1 次
re.findall('年.+', text2)  # "年" 后面可以跟任意多个字符
re.findall('年+.', text2)  # "年" 可以出现 1 次或多次，后面跟任意 1 个字符
re.findall('年.?', text2)  # "年" 后面至多可以跟 1 个任意字符
re.findall('年.*', text2)  # "年" 后面可以跟任意多个字符
re.findall('年.+?', text2)  # "年" 后面可以跟一个任意字符，并且这些任意字符至多出现 1 次
re.findall('年.*?', text2)  # "年" 后面允许不带其他字符的内容
re.findall('年?花', text2)  # "花" 前面的 "年" 至多出现 1 次
re.findall('年*花', text2)  # "花" 前面的 "年" 可以出现 0 次、1 次或多次
re.findall('年+花', text2)  # "花" 前面的 "年" 可以出现 1 次或多次
re.findall('年{1}花', text2)  # "花" 前面的 "年" 出现 1 次
re.findall('年{2}花', text2)  # "花" 前面的 "年" 出现 2 次
re.findall('年{0,1}花', text2)  # "花"前面的"年"出现0次或1次
re.findall('年.+?花', text2)  # "年"开头、"花"结尾且中间可以有任意多个任意字符
re.findall('年.*?花', text2)  # "年"开头、"花"结尾且中间至多有一个任意字符
re.findall('年.*花', text2)  # "年"开头、"花"结尾且中间可以有任意多个任意字符
re.findall('年.+?花', text2)  # "年"开头、"花"结尾且中间至少有一个任意字符
re.findall('年.*?花', text2)  # "年"开头、"花"结尾中间允许不带其他字符的内容

# --- 代码 3-10：字符集 [] 匹配括号内的任一字符 ---
#3-10
print(re.findall('[科数]学',text1))

# --- 代码 3-11：锚点 ^ 匹配字符串开头 ---
#3-11
p_string = text1.split('。')
for line in p_string:
    if len(re.findall('^自', line)):
        print(line)

# --- 代码 3-12：锚点 $ 匹配字符串结尾 ---
#3-12
p_string = text1.split('、')
for line in p_string:
    if len(re.findall('学$', line)):
        print(line)

# --- 代码 3-13：预定义字符集 \d \s \w \b ---
#3-13
text3 = 'Hello, everyone, 我是/陈_X/ 我的_/、邮箱，地址是。wxid_6cp816.co'
re.sub('\\d', '数字', text3)
re.sub(r'\d', '数字', text3)
re.sub('[0-9]', '数字', text3)
re.sub(r'\s', '', text3)
re.sub(r'\W', '', text3)
re.findall(r'\\b[a-zA-Z]+\\b', text3)
re.findall(r'\\b[a-zA-Z]+\\b', text3)

# 代码3-14
print(re.findall('自.语言处理', text1))

# 代码3-15
print(re.findall('方法|计算机', text1))

# 代码3-16
p_string = text1.split('。')
for line in p_string:
    if re.search('方法|计算机', line):
        print(line)


