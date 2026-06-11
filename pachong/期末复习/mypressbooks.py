# -*- coding: utf-8 -*-
"""
当当网图书信息爬虫 —— 按出版社搜索并抓取图书详情（原始版）。

教学重点：
- 从 txt 文件按行读取出版社名称列表
- requests + BeautifulSoup 抓取当当网高级搜索页面
- urllib.parse.urlencode 将中文参数编码并拼接到 URL
- BeautifulSoup 的 select / select_one 逐层定位图书元素
- 数据写入本地 txt 文件
"""

# --- 1. 导入模块 ---
import requests
from bs4 import BeautifulSoup
import traceback
import os
# 引入BrLiD库
import urllib.parse#题目3

# --- 2. 读取出版社列表 ---

# 读取出版社列表
def read_list(txt_path):#题目(2)
    press_list = []
    f = open(txt_path, 'r', encoding='utf-8')
    for line in f.readlines():
        press_list.append(line.strip())# 题目(1)添加出版社
    return press_list

# --- 3. 构造搜索URL ---

# 定位<input>标签，拼接URL
def build_form(press_name):
    header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko'}
    res = requests.get(f'http://search.dangdang.com/advsearch', headers=header)  # 请求当当高级搜索页
    res.encoding = 'gb2312'  # 设置编码为 gb2312，适配当当网页面
    soup = BeautifulSoup(res.text, 'html.parser')  # 解析 HTML
    # 定位<input>标签
    input_tag_name = ''
    conditions = soup.select('.box2 > .detail_condition > label')  # CSS 选择器：定位搜索条件区域
    print('共找到 %d 项基本条件，正在寻找<input>标签：' % len(conditions))
    for item in conditions:
        text = item.select('span')[0].string  # 获取条件标签文本
        if text == '出版社':
            input_tag_name = ''
            print('已经找到<input>标签，name:', input_tag_name)

    # 拼接URL
    keyword = {
        'medium': '01',  # 01 代表图书类别
        'input_tag_name': press_name.encode('gb2312'),  # 出版社名编码为 gb2312
        'category_path': '01.00.00.00.00.00',  # 图书分类路径
        'sort_type': 'sort_score_desc'  # 按评分降序排序
    }
    url = 'http://search.dangdang.com/'
    # 拼接编码，拼接url
    url += urllib.parse.urlencode(keyword)#题目(4)  # 将参数字典编码为 URL 查询字符串
    print('入口地址：%s' % url)
    return

# --- 4. 抓取图书信息 ---

# 抓取信息，参考图8-7的图书记录页面的HTML源代码中的相关字段标签
def get_info(entry_url):
    header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko'}
    res = requests.get(entry_url, headers=header)  # 请求入口 URL
    res.encoding = 'gb2312'
    soup = BeautifulSoup(res.text, 'html.parser')
    # 获取页数（保持不变）
    page_num = int(soup.select_one('div.data span').text.strip())  # 提取总页数
    print('共 %d 页待抓取，这里只测试采集1页' % page_num)
    page_now = 1  # 测试只抓1页
    page_now = '&page_index='  # 分页参数名

    # 初始化列表（保持不变）
    books_title = []  # 存放书名
    books_price = []  # 存放价格
    books_date = []  # 存放出版日期
    books_comment = []  # 存放评论数

    for i in range(1, page_num + 1):  # 遍历每一页
        now_url = entry_url + page_now + str(i)  # 拼接分页 URL
        print('正在获取第 %d 页，URL: %s' % (i, now_url))
        res = requests.get(now_url, headers=header)
        res.encoding = 'gb2312'  # 这里补充编码，避免解析乱码
        soup = BeautifulSoup(res.text, 'html.parser')

        # 关键修改：先获取当前页所有图书条目（每个1对应一本书）
        book_items = soup.select('li.bigimg > ul[dt-pit]')  # 定位每个图书条目

        # 逐个处理每本书
        for li in book_items:
            # 1. 提取书名
            title_tag = li.select_one('a')  # 每本书的标题标签
            title = title_tag.get('title') if title_tag else ''  # 防止标签不存在
            books_title.append(title)

            # 2. 提取价格
            price_tag = li.select_one('p.price > span.search_now_price')
            price = price_tag.text if price_tag else ''
            books_price.append(price)

            # 3. 提取评论数量
            # 这里可以添加提取评论数量的代码
            comment_tag = li.select_one('p.search_star_line > a')
            comment = comment_tag.text.replace("条评论","")#题目(5)
            books_comment.append(comment)

            # 4. 提取出版日期（核心修改）
            # 每个li下的p.search_book_author包含多个span（作者、出版社、日期等）
            author_spans = li.select('p.search_book_author > span')  # 获取作者信息区域的所有 span
            date = ''  # 默认空值
            if len(author_spans) >= 3:
                date = author_spans[2].text[2:]  # 取第3个span，去掉前两个字符，保持一致
            books_date.append(date)  # 每本书添加一个日期（空值或实际值）

    # 构建字典返回(保持不变)
    # 构建字典返回(保持不变)
    books_dict = {
        'title': books_title,
        'price': books_price,
        'date': books_date,
        'comment': books_comment
    }
    return books_dict

# --- 5. 保存数据 ---

# 保存数据
def save_info(file_dir, press_name, books_dict):
    res = ''
    try:
        for i in range(len(books_dict['title'])):
            res += (str(i+1) + '、' + '书名：' + books_dict['title'][i] + '\r\n' +
                    '价格：' + books_dict['price'][i] + '\r\n' +
                    '出版日期：' + books_dict['date'][i] + '\r\n' +
                    '评论数量：' + books_dict['comment'][i] + '\r\n' +
                    '\r\n')
    except Exception as e:
        print('保存出错')
        print(e)
        traceback.print_exc()
    finally:
        file_path = file_dir + os.sep + press_name + '.txt'  # 拼接保存文件的完整路径
        f = open(file_path, 'w')
        f.write(res.encode('utf-8'))
        # 关闭文档
        f.close()
        return

# --- 6. 主入口函数 ---

# 入口
def start_spider(press_path, saved_file_dir):
    # 获取出版社列表
    press_list = read_list(press_path)  # 读取出版社列表
    for press_name in press_list:
        print('------- 开始抓取 %s -------' % press_name)
        press_page_url = build_form(press_name)  # 构造搜索 URL
        books_dict = get_info(press_page_url)  # 抓取图书数据
        save_info(saved_file_dir, press_name, books_dict)  # 保存到文件
        print('------- 出版社: %s 抓取完毕 -------' % press_name)
    return

# --- 7. 程序入口 ---

if __name__ == '__main__':
    # 出版社列表文件路径
    press_txt_path = r'press.txt'  # 出版社名称列表文件
    # 抓取信息保存路径
    saved_file_dir = r'D:'  # 保存目录
    # 启动
    start_spider(press_txt_path, saved_file_dir)  # 启动爬虫



