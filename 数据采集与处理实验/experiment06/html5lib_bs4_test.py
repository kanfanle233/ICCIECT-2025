"""
综合演示 html5lib + BeautifulSoup + lxml 的 HTML 解析方法。

教学重点：
- html5lib.parse() 将 HTML 解析为 lxml Element 树并使用 XPath
- BeautifulSoup 的 DOM 遍历：children、descendants
- BeautifulSoup 的 findAll() 和 select() 方法（CSS 选择器）
"""

# --- 1. 导入模块 ---
import html5lib
import requests
import chardet
import bs4

# --- 2. 获取并解析百度首页 ---
resp = requests.get(url="http://www.baidu.com/")
html = resp.content
html = html.decode(chardet.detect(html)['encoding'])  # 自动检测编码
bs = bs4.BeautifulSoup(html, features='lxml')
html = bs.prettify()  # 格式化 HTML
print(html)

# --- 3. 使用 html5lib 解析 + lxml xpath 提取超链接 ---
print("-" * 100)
print('显示所有<a>标签中的超链接和显示文本')
element = html5lib.parse(html, treebuilder='lxml', namespaceHTMLElements=False)  # html5lib 解析为 lxml 树
print('html5lib.parse()返回的是', type(element))

anchors = element.xpath('//a')  # XPath 选取所有 <a> 节点
for anchor in anchors:
    print("<a href='%s'>%s</a>" % (anchor.attrib['href'], anchor.text))

# --- 4. BeautifulSoup DOM 遍历方式 ---
print('_' * 30 + '使用BeautifulSoup获取文档标签和内容' + '_' * 30)
print('文档标题：', bs.html.head.title.text)        # 通过标签链导航
print('body的Link属性：', bs.html.body['link'])      # 获取标签属性
print('body的子标签：')
for child in bs.html.body.children:                   # 直接子节点（仅一层）
    print('\t', child.name)
print('body中的所有a标签：')
for anchor in bs.html.body.descendants:               # 所有后代节点（递归）
    if anchor.name == 'a':
        print("\t<a href='%s'>%s</a>" % (anchor['href'], anchor.text))

# --- 5. BeautifulSoup findAll 方法 ---
print('_' * 30 + '使用BeautifulSoup的findAll方法' + '_' * 30)
for anchor in bs.html.body.findAll(name='a'):         # 按标签名查找所有匹配元素
    if anchor.name == 'a':
        print("\t<a href='%s'>%s</a>" % (anchor['href'], anchor.text))

# --- 6. BeautifulSoup select 方法（CSS 选择器） ---
print('_' * 30 + '使用BeautifulSoup的select方法' + '_' * 30)
print('所有的a标签：', bs.select('a'))                        # 标签选择器
print('所有的class属性等于或包含mnav的标签：', bs.select('.mnav'))  # 类选择器
print('id=ftCon的标签：', bs.select('#ftCon'))                  # ID 选择器
print('id=Lh的p标签：', bs.select('p#Lh'))                     # 标签 + ID 组合
print('所有的class属性等于或包含mnav的a标签：', bs.select('a.mnav'))  # 标签 + 类组合
print('name属性值等于tj_trnews的a标签：', bs.select('a[name="tj_trnews"]'))  # 属性选择器
print('所有class属性等于或包含mnav的a标签，且其父标签是div：', bs.select('div a.mnav'))  # 后代选择器

print('所有的a标签：')
for anchor in bs.select('a'):
    print("\t<a href='%s'>%s</a>" % (anchor['href'], anchor.text))
