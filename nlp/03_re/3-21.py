"""
正则表达式提取结构化信息。

教学重点：使用 re.search 的分组捕获功能
从固定格式的文本中提取姓名、性别、年龄等字段。
"""

import re

# --- 1. 定义待解析的结构化文本 ---
text = """
姓名：张三
性别：男
年龄：25岁
电话号码：13512345678
地址：北京市朝阳区
"""

# --- 2. 定义正则模式（使用 .*? 非贪婪匹配捕获每个字段值） ---
name_pattern = r"姓名：(.*?)\n"
gender_pattern = r"性别：(.*?)\n"
age_pattern = r"年龄：(.*?)\n"
phone_pattern = r"电话号码：(.*?)\n"
address_pattern = r"地址：(.*?)\n"

# --- 3. 逐一提取并输出 ---
name = re.search(name_pattern, text).group(1)
gender = re.search(gender_pattern, text).group(1)
age = re.search(age_pattern, text).group(1)
phone = re.search(phone_pattern, text).group(1)
address = re.search(address_pattern, text).group(1)

print("姓名：", name)
print("性别：", gender)
print("年龄：", age)
print("电话号码：", phone)
print("地址：", address)
