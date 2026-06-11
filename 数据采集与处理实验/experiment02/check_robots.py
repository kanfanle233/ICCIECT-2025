"""
获取并查看网站的 robots.txt 文件内容。

教学重点：
- robots.txt 是网站对爬虫的访问规则声明
- 通过 requests 获取并用 chardet 解码
"""

# --- 1. 导入模块 ---
import requests
import chardet

# --- 2. 获取百度的 robots.txt ---
res = requests.get("http://www.baidu.com/robots.txt")
s = res.content.decode(chardet.detect(res.content)['encoding'])  # 自动检测编码并解码
print(s)