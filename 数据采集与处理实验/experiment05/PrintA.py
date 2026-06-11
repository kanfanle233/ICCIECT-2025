"""
使用 HTMLParser 提取百度首页中所有 <a> 标签的超链接地址。

教学重点：
- 自定义 HTMLParser 子类，仅处理 <a> 标签
- 结合 requests + chardet 获取并解码页面
- BeautifulSoup 的 prettify() 可辅助标准化 HTML
"""

# --- 1. 导入模块 ---
from html.parser import HTMLParser
import requests
from chardet import detect
import bs4

# --- 2. 自定义解析器：只提取 <a> 标签 ---
class PrintA(HTMLParser):
    """提取并打印所有 <a> 标签的 href 属性值。"""

    def handle_starttag(self, tag, attrs):
        """遇到开始标签时回调，仅处理 <a> 标签。"""
        if tag == 'a':
            for attr, value in attrs:
                if attr == 'href':
                    print('a:', value)  # 打印超链接地址
                    break

# --- 3. 获取页面并解析 ---
if __name__ == '__main__':
    res = requests.get("http://www.baidu.com")
    if res.status_code == 200:
        html = res.content.decode(detect(res.content)['encoding'])  # 自动检测编码
        bs = bs4.BeautifulSoup(html)
        html = bs.prettify()  # 格式化 HTML，便于解析
        print(html)
        print_a = PrintA()
        print_a.feed(html)  # 将 HTML 喂给解析器
    else:
        print('Found error:', res.status_code)