"""
使用 PyQuery（jQuery 风格的 Python 库）解析 HTML 并提取元素。

教学重点：
- PyQuery 使用 CSS 选择器语法（类似 jQuery）
- 与 BeautifulSoup 的 prettify() 配合使用
- 遍历 PyQuery 对象获取元素属性和文本
"""

# --- 1. 导入模块 ---
import pyquery
import requests
import chardet
import bs4

# --- 2. 获取并预处理页面 ---
resp = requests.get(url="http://www.baidu.com/")
html = resp.content
html = html.decode(chardet.detect(html)['encoding'])  # 自动检测编码
bs = bs4.BeautifulSoup(html, features='lxml')
html = bs.prettify()  # 格式化 HTML
print(html)

# --- 3. 使用 PyQuery 的 CSS 选择器提取元素 ---
print('-' * 30 + '使用PyQuery' + '-' * 30)
pq = pyquery.PyQuery(html)  # 用 HTML 字符串创建 PyQuery 对象
for a in pq('body div a.mnav'):  # CSS 选择器：body 下 div 下 class 含 mnav 的 <a>
    print('<a class="%s" href="%s">%s</a>' % (
        a.attrib['class'],   # 获取 class 属性
        a.attrib['href'],    # 获取 href 属性
        a.text.strip()       # 获取并清理文本内容
    ))