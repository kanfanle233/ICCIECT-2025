"""
使用 requests 获取百度首页，演示编码检测、保存和 BeautifulSoup 解析。

教学重点：
- requests.get() 获取网页及响应对象属性
- chardet 自动检测网页编码
- 二进制模式保存网页到本地文件
- BeautifulSoup 格式化输出 HTML
"""

# --- 1. 导入模块 ---
import chardet
import requests
from bs4 import BeautifulSoup

# --- 2. 发起请求并查看响应信息 ---
res = requests.get("https://www.baidu.com")
print(res.ok)           # 请求是否成功（状态码 < 400）
print(res.content)      # 响应的二进制内容

# --- 3. 编码检测与解码 ---
print(res.encoding)                    # requests 推测的编码
print(chardet.detect(res.content))     # chardet 检测的编码（含置信度）

encoding = chardet.detect(res.content)['encoding']  # 取出检测到的编码
s = res.content.decode(encoding)                    # 用检测到的编码解码
print(s)

# --- 4. 保存网页到本地文件 ---
with open("baidu.html", "wb") as f:    # 二进制写入
    f.write(res.content)

# --- 5. BeautifulSoup 解析并美化输出 ---
soup = BeautifulSoup(s, features="lxml")
print(soup.prettify())
