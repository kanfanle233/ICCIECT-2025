"""
复旦大学新闻列表爬虫 —— 使用 requests + BeautifulSoup 抓取新闻信息。

教学重点：
- requests 发送 HTTP GET 请求并设置请求头
- BeautifulSoup 的 CSS 选择器（select）定位嵌套元素
- 提取标签属性（title、href）和文本内容
- 状态码判断（response.status_code）
- 追加模式写入 CSV 文件
"""

# --- 1. 导入模块 ---
# 导入相关库
import requests
from bs4 import BeautifulSoup

# --- 2. 发送请求 ---

url = "https://news.fudan.edu.cn/xxyw/list.htm"  # 复旦大学校内新闻列表页 URL
myhead = {
    'User-Agent': 'Mozilla/5.0',  # 请求头，伪装为浏览器访问
}
response = requests.get(url, headers=myhead)  # 发送 GET 请求
response.encoding = "utf-8"  # 设置响应编码为 UTF-8，防止中文乱码

# --- 3. 解析页面并提取数据 ---

if response.status_code == 200:  # 状态码 200 表示请求成功
    print(response.text[:10])  # 打印前 10 个字符，用于调试验证
    soup = BeautifulSoup(response.text, 'html.parser')#3分  # 用 BeautifulSoup 解析 HTML
    news_list = soup.select(".news_box .news_con")  # CSS 选择器：定位新闻容器，class 同时包含 news_box 和 news_con
    print(len(news_list))  # 打印新闻条数，用于验证选择器是否正确

    for news in news_list:  # 遍历每条新闻
        title = news.select(".news_title a")[0].get("title")  # 提取新闻标题（a 标签的 title 属性）
        href = news.select(".news_title a")[0].get("href")  # 提取新闻链接（a 标签的 href 属性）
        content = news.select(".news_text")[0].text  # 提取新闻摘要文本
        date = news.select(".news_time .times")[0].text  # 提取发布日期
        view = news.select(".news_time .views .wp_listVisitCount")[0].text  # 提取浏览量
        print("方昕哲202314109",title,href,content,date,view)  # 打印提取结果

        # --- 4. 保存到 CSV 文件 ---
        with open("fudan_out.csv","a",encoding="utf-8-sig") as file:  # 追加模式写入 CSV，utf-8-sig 编码可在 Excel 中正确显示中文
            file.write(title+","+href+","+content+","+date+","+view+"\n")  # 以逗号分隔写入一行



