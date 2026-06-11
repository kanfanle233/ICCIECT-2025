"""
从百度首页中提取所有 URL 链接。

教学重点：
- requests 获取页面 + chardet 解码 + BeautifulSoup 格式化
- 正则表达式匹配 HTTP URL（支持大小写字母）
"""

# --- 1. 导入模块 ---
import requests
import chardet
import re
import bs4

# --- 2. 获取并解析页面 ---
res = requests.get('http://www.baidu.com')
s = res.content.decode(chardet.detect(res.content)['encoding'])  # 自动检测编码
s = bs4.BeautifulSoup(s, features='lxml').prettify()             # 格式化 HTML
print(s)

# --- 3. 用正则提取所有 HTTP URL ---
flag = re.IGNORECASE  # 忽略大小写
for url in re.findall('http://[a-z0-9/.,A-Z]*', s, flag):  # 匹配字母、数字和常见 URL 字符
    print(url)