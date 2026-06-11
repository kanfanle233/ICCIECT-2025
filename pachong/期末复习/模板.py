"""
复旦大学主页爬虫模板 —— 最简爬虫框架，供课堂练习参考。

教学重点：
- requests 发送最基础的 HTTP GET 请求
- BeautifulSoup 解析网页的基本流程（导入 -> 请求 -> 解析 -> 提取）
- 此文件为练习模板，后续可在 if response.status_code == 200 块中添加解析逻辑
"""

# --- 1. 导入模块 ---
import requests  # 用于发送 HTTP 请求
from bs4 import BeautifulSoup  # 用于解析 HTML 页面

# --- 2. 发送请求 ---

url = "https://www.fudan.edu.cn/"  # 复旦大学主页 URL
# 注意：此处省略了请求头设置和 response 的获取，可在下方自行补充完整逻辑
# response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
# response.encoding = 'utf-8'
# soup = BeautifulSoup(response.text, 'html.parser')
