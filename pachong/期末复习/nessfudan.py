"""
当当网图书信息爬虫（精简版）—— 按出版社搜索，提取书名、价格、出版日期和评论数。

教学重点：
- 从文件读取出版社列表并循环处理
- requests + urllib.parse 构造带中文参数的搜索 URL
- BeautifulSoup 的 find / find_all 定位图书元素
- 字符串分割提取出版日期（split 方法）
- 三元表达式处理空值（if ... else "N/A"）
"""

# --- 1. 导入模块 ---
import requests
from bs4 import BeautifulSoup
import urllib.parse

# --- 2. 读取出版社列表 ---

def read_list(txt_path):
    """从 txt 文件按行读取出版社名称"""
    press_list = []
    f = open(txt_path, "r", encoding='utf-8')
    for line in f.readlines():
        press_list.append(line.strip('\n'))  # 去除行末换行符
    return press_list

# --- 3. 构造搜索参数并发送请求 ---

press_txt_path = r'press.txt.txt'  # 出版社列表文件路径（注意文件名）
presslist = read_list(press_txt_path)  # 读取所有出版社名称
print(presslist)  # 打印出版社列表，用于调试

for apress in presslist:  # 遍历每个出版社
    print('------开始抓取 %s------' % apress)

    url = "http://search.dangdang.com/advsearch "  # 当当网高级搜索入口 URL
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko'}  # 请求头

    # input_tag_name = ""  # 以下为注释掉的<input>标签定位代码（备用方案）
    # conditions = soup.select('.box2 > .detail_condition > label')
    # print('共找到%d项基本条件，正在寻找<input>标签' % len(conditions))
    # for item in conditions:
    #    text = item.select('span')[0].string
    #    if text == '出版社':
    #        input_tag_name = item.select('input')[0].get('name')
    #        print('已经找到<input>标签，name：', input_tag_name)

    # --- 4. 构造搜索参数 ---

    myparameters = {
        "medium": "01",  # 01 代表图书
        "key3": apress.encode('gb2312'),  # 出版社名称编码为 gb2312 格式
        "category_path": "01.00.00.00.00.00"  # 图书分类路径
        #"sort_type" : 'sort_score_desc'  # 注释掉的排序参数
    }

    url = 'http://search.dangdang.com/?'  # 搜索结果页基础 URL
    url = url + urllib.parse.urlencode(myparameters)  # 将参数字典编码为 URL 查询字符串
    print('入口地址： %s' % url)

    # --- 5. 发送请求并解析页面 ---

    response = requests.get(url, headers=headers, params=myparameters)  # 发送 GET 请求（注意：params 和手动拼接 URL 同时使用）
    response.encoding = 'gb2312'  # 设置编码为 gb2312
    print(response.text[:10])  # 打印前 10 个字符，用于调试

    soup = BeautifulSoup(response.text, 'html.parser')  # 解析 HTML 页面

    # --- 6. 提取图书信息 ---

    # 书名，价格，出版日期，评论数量。
    mybooks = soup.find("ul", "bigimg").find_all("li")  # 定位图书列表（class="bigimg" 的 ul 下的所有 li）

    for item in mybooks:  # 遍历每本图书
        # 分别定位书名、价格、作者信息、评论数元素
        name_elem = item.find("p", class_="name")  # 书名标签
        price_elem = item.find("span", class_="price_n")  # 价格标签
        author_elem = item.find("p", class_="search_book_author")  # 作者/出版社信息标签
        comment_elem = item.find("a", class_="search_comment_num")  # 评论数标签

        # 提取文本，若元素不存在则用 "N/A" 代替
        name = name_elem.text if name_elem else "N/A"  # 书名
        price = price_elem.text if price_elem else "N/A"  # 价格
        pub_date = author_elem.text.split('/')[-2].strip() if author_elem and '/' in author_elem.text else "N/A"  # 出版日期：用 "/" 分割后取倒数第二段
        comment = comment_elem.text if comment_elem else "N/A"  # 评论数量

        print('书名:%s | 价格:%s | 出版日期:%s | 评论:%s' % (name, price, pub_date, comment))
        break  # 每个出版社只打印第一条，用于测试
