"""
使用 lxml 的 XPath 语法提取 HTML 中的超链接。

教学重点：
- lxml.etree.fromstring() 解析 HTML 字符串
- XPath 表达式 //a 选取所有 <a> 节点
- 通过 attrib 和 text 属性获取元素的属性值和文本内容
"""

# --- 1. 导入模块 ---
from lxml import etree

# --- 2. 准备 HTML 内容 ---
html = '''
<html>
<body>
<a href="http://www.baidu.com">BaiDu</a>
<a href="http://www.sohu.com">Sohu</a>
<a href="http://www.qq.com">QQ</a>
</body></html>
'''

# --- 3. 解析并用 XPath 提取 <a> 标签 ---
elem = etree.fromstring(html)  # 将 HTML 字符串解析为 Element 对象
all_a = elem.xpath('//a')       # XPath：选取文档中所有 <a> 节点
for a in all_a:
    print("a:", a.attrib['href'], a.text)  # 获取 href 属性和链接文本