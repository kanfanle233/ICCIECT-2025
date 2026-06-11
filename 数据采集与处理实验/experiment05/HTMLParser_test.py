"""
使用标准库 HTMLParser 解析 HTML，以缩进格式打印 DOM 树。

教学重点：
- html.parser.HTMLParser 的事件驱动解析模型
- 重写 handle_starttag / handle_endtag / handle_data 回调方法
- 通过 depth 变量实现缩进层级控制
"""

# --- 1. 导入模块 ---
from html.parser import HTMLParser

# --- 2. 自定义 HTML 解析器 ---
class MyParser(HTMLParser):
    """带缩进输出的 HTML 解析器，继承自 HTMLParser。"""

    def __init__(self):
        super().__init__()
        self.depth = 0  # 当前嵌套深度，用于控制缩进

    def handle_starttag(self, tag, attrs):
        """处理开始标签，打印标签名和属性，并增加嵌套深度。"""
        print('%s<%s' % ('\t' * self.depth, tag), end='')
        for name, value in attrs:
            print(' %s="%s"' % (name, value), end='')  # 打印属性名="属性值"
        print('>')
        self.depth += 1

    def handle_endtag(self, tag):
        """处理结束标签，减少嵌套深度并打印闭合标签。"""
        self.depth -= 1
        print('%s</%s>' % ('\t' * self.depth, tag))

    def handle_data(self, data):
        """处理标签之间的文本内容。"""
        print('%s%s' % ('\t' * self.depth, data))

# --- 3. 使用解析器 ---
if __name__ == '__main__':
    parser = MyParser()
    parser.feed("""
    <html><head><title>Hello</title></head>
    <body><h3 align="center">Hello world!</h3></body>
    </html>
    """)