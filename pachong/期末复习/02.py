"""
豆瓣电影Top250爬虫（初版/练习）—— 使用 requests + BeautifulSoup 抓取电影信息。

教学重点：
- requests 发送 HTTP 请求访问豆瓣页面
- BeautifulSoup 的 CSS 选择器定位电影条目
- 状态码判断（response.status_code == 200）
- 注意：此脚本为课堂练习初版，仅能抓取第一页，且代码未完成
"""

# --- 1. 导入模块 ---
import requests
from bs4 import BeautifulSoup

# --- 2. 发送请求 ---

url = "https://movie.douban.com/top250"  # 豆瓣电影 Top250 首页 URL
headers = {"User-Agent": "Mozilla/5.0"}  # 请求头，伪装为浏览器
response = requests.get(url, headers=headers)  # 发送 GET 请求
response.encoding = "utf-8"  # 设置响应编码

# --- 3. 解析页面并提取数据 ---

if response.status_code == 200:  # 请求成功
    print(response.text[:10])  # 打印前 10 个字符，用于调试
    soup = BeautifulSoup(response.text, 'html.parser')#3分  # 解析 HTML 页面
    title_list = soup.select(".item .info")  # CSS 选择器：定位每个电影条目（class="item" 下的 class="info"）
    print(len(title_list))  # 打印条目数量，验证选择器

    for title in title_list:  # 遍历每部电影（注意：此循环未完成，缺少提取逻辑）
        title = title.selec()  # 此处调用不完整，selec() 缺少参数，属于课堂练习中的遗留代码



