"""
网页解析练习脚本。

教学重点：从网页结构中提取需要的文本内容。
"""

import requests
from bs4 import BeautifulSoup

# 创建会话，自动管理 Cookie / Session
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36"
})

# 请求首页
url = "https://www.kactusbio.cn/"
resp = session.get(url)

print("状态码:", resp.status_code)
print("Session:", session.cookies.get_dict())

# 解析 HTML
soup = BeautifulSoup(resp.text, "html.parser")

# 示例：查找产品标题与链接（需根据页面结构调整选择器）
for a in soup.select("a[href*='product']"):
    title = a.text.strip()
    link = a["href"]
    if title and link:
        if not link.startswith("http"):
            link = "https://www.kactusbio.cn/" + link.lstrip("/")
        print(title, "→", link)
