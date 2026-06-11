# -*- coding: utf-8 -*-
"""
当当网图书信息爬虫 —— 按出版社搜索并抓取图书详情（详细注释版）。

教学重点：
- 从 txt 文件读取出版社列表，循环遍历爬取
- requests 发送 GET 请求并处理 gb2312 编码
- BeautifulSoup 的 CSS 选择器（select、select_one）定位复杂页面元素
- urllib.parse.urlencode 拼接带中文参数的 URL
- 数据格式化写入本地 txt 文件
"""

# --- 1. 导入模块 ---
import requests  # 用于发送网络请求的库
from bs4 import BeautifulSoup  # 用于解析HTML网页结构的库
import traceback  # 输出详细错误堆栈信息
import os  # 处理文件路径相关操作
# 引入BrLiD库
import urllib.parse  # 用于URL编码 #题目3

# --- 2. 读取出版社列表 ---

# 读取出版社列表
def read_list(txt_path):  # 题目(2)
    press_list = []  # 初始化存放出版社名字的列表
    f = open(txt_path, 'r', encoding='utf-8')  # 打开出版社列表文件
    for line in f.readlines():  # 逐行读取
        press_list.append(line.strip())  # 去除换行符并加入列表 # 题目(1)添加出版社
    return press_list  # 返回出版社名称列表

# --- 3. 构造搜索URL ---

# 定位<input>标签，拼接URL
def build_form(press_name):
    header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko'}
    # 请求当当高级搜索页面，获取页面结构
    res = requests.get(f'http://search.dangdang.com/advsearch', headers=header)
    res.encoding = 'gb2312'  # 设置网页解析编码为gb2312
    soup = BeautifulSoup(res.text, 'html.parser')  # 解析HTML页面

    # 定位<input>标签
    input_tag_name = ''  # 存储<input>标签的name属性
    conditions = soup.select('.box2 > .detail_condition > label')  # 选择所有基本搜索条件的label标签
    print('共找到 %d 项基本条件，正在寻找<input>标签：' % len(conditions))
    for item in conditions:
        text = item.select('span')[0].string  # 获取标签中的文本内容
        if text == '出版社':  # 找到"出版社"输入框对应区域
            input_tag_name = ''  # 此处未实际解析 input 的 name，仅占位
            print('已经找到<input>标签，name:', input_tag_name)

    # 拼接URL
    keyword = {
        'medium': '01',  # medium 参数，固定值，代表图书
        'input_tag_name': press_name.encode('gb2312'),  # 将出版社名称编码为 gb2312 格式
        'category_path': '01.00.00.00.00.00',  # 分类路径
        'sort_type': 'sort_score_desc'  # 按评分排序
    }
    url = 'http://search.dangdang.com/'
    # 拼接编码，拼接url
    url += urllib.parse.urlencode(keyword)  # 将参数字典编码为URL格式字符串 #题目(4)
    print('入口地址：%s' % url)
    return url  # 返回完整入口URL（原代码未返回，这里保持你的原逻辑，但此处应返回）

# --- 4. 抓取图书信息 ---

# 抓取信息，参考图8-7的图书记录页面的HTML源代码中的相关字段标签
def get_info(entry_url):
    header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko'}
    res = requests.get(entry_url, headers=header)  # 请求出版社入口URL
    res.encoding = 'gb2312'  # 设置编码
    soup = BeautifulSoup(res.text, 'html.parser')  # 解析页面内容

    # 获取页数（保持不变）
    page_num = int(soup.select_one('div.data span').text.strip())  # 提取总页数
    print('共 %d 页待抓取，这里只测试采集1页' % page_num)
    page_now = 1  # 测试只抓1页
    page_now = '&page_index='  # 实际分页参数名

    # 初始化列表（保持不变）
    books_title = []  # 书名列表
    books_price = []  # 价格列表
    books_date = []  # 出版日期列表
    books_comment = []  # 评论数列表

    for i in range(1, page_num + 1):  # 遍历每一页
        now_url = entry_url + page_now + str(i)  # 构造当前页URL
        print('正在获取第 %d 页，URL: %s' % (i, now_url))
        res = requests.get(now_url, headers=header)  # 请求每页
        res.encoding = 'gb2312'  # 这里补充编码，避免解析乱码
        soup = BeautifulSoup(res.text, 'html.parser')

        # 关键修改：先获取当前页所有图书条目（每个li对应一本书）
        book_items = soup.select('li.bigimg > ul[dt-pit]')  # 提取每本书所在的ul标签

        # 逐个处理每本书
        for li in book_items:
            # 1. 提取书名
            title_tag = li.select_one('a')  # 定位标题链接
            title = title_tag.get('title') if title_tag else ''  # 获取书名（处理空值）
            books_title.append(title)

            # 2. 提取价格
            price_tag = li.select_one('p.price > span.search_now_price')  # 现价标签
            price = price_tag.text if price_tag else ''  # 提取价格文本
            books_price.append(price)

            # 3. 提取评论数量
            comment_tag = li.select_one('p.search_star_line > a')  # 评论链接
            comment = comment_tag.text.replace("条评论", "")  # 去掉"条评论"字样 #题目(5)
            books_comment.append(comment)

            # 4. 提取出版日期
            author_spans = li.select('p.search_book_author > span')  # 获取作者出版社日期等span
            date = ''  # 默认空值
            if len(author_spans) >= 3:  # 第三个span是出版日期
                date = author_spans[2].text[2:]  # 去除前两个字符，例如" / "
            books_date.append(date)  # 保存日期

    # 构建字典返回(保持不变)
    books_dict = {
        'title': books_title,
        'price': books_price,
        'date': books_date,
        'comment': books_comment
    }
    return books_dict  # 返回所有数据

# --- 5. 保存数据 ---

# 保存数据
def save_info(file_dir, press_name, books_dict):
    res = ''  # 初始化保存字符串
    try:
        for i in range(len(books_dict['title'])):  # 遍历每本书
            res += (str(i+1) + '、' + '书名：' + books_dict['title'][i] + '\r\n' +
                    '价格：' + books_dict['price'][i] + '\r\n' +
                    '出版日期：' + books_dict['date'][i] + '\r\n' +
                    '评论数量：' + books_dict['comment'][i] + '\r\n' +
                    '\r\n')  # 拼接每本书的信息
    except Exception as e:  # 捕捉异常
        print('保存出错')
        print(e)
        traceback.print_exc()  # 打印错误堆栈
    finally:
        file_path = file_dir + os.sep + press_name + '.txt'  # 构造文件路径
        f = open(file_path, 'w')  # 打开文件
        f.write(res.encode('utf-8'))  # 写入内容（注：这里实际会报错，因为 write() 不能写 bytes）
        # 关闭文档
        f.close()
        return

# --- 6. 主入口函数 ---

# 入口
def start_spider(press_path, saved_file_dir):
    # 获取出版社列表
    press_list = read_list(press_path)  # 读取出版社名称
    for press_name in press_list:
        print('------- 开始抓取 %s -------' % press_name)
        press_page_url = build_form(press_name)  # 构造出版社搜索入口
        books_dict = get_info(press_page_url)  # 抓取图书信息
        save_info(saved_file_dir, press_name, books_dict)  # 保存数据
        print('------- 出版社: %s 抓取完毕 -------' % press_name)
    return

# --- 7. 程序入口 ---

if __name__ == '__main__':
    # 出版社列表文件路径
    press_txt_path = r'press.txt'  # 存放出版社名称的文本文件
    # 抓取信息保存路径
    saved_file_dir = r'D:'  # 保存结果的文件夹路径
    # 启动
    start_spider(press_txt_path, saved_file_dir)  # 调用入口函数启动爬虫



