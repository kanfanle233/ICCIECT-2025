"""
综合示例：requests 获取页面 + chardet 检测编码 + BeautifulSoup 格式化 + 正则提取 URL。

教学重点：
- requests 获取网页二进制内容
- chardet 自动检测网页编码
- BeautifulSoup 对 HTML 进行美化输出
- 正则表达式从 HTML 中提取所有 URL
"""

# --- 1. 导入模块 ---
import requests
import chardet
import re
import bs4

# --- 2. 获取页面并解码 ---
res = requests.get('http://www.baidu.com')  # 发起 HTTP GET 请求
s = res.content.decode(chardet.detect(res.content)['encoding'])  # 自动检测编码并解码
s = bs4.BeautifulSoup(s, features='lxml').prettify()  # 用 lxml 解析并美化 HTML
print(s)

# --- 3. 用正则提取所有 URL ---
flag = re.IGNORECASE  # 忽略大小写
for url in re.findall('http://[a-z,0-9,/,.,_]*', s, flag):
    print(url)
