"""
正则与 BeautifulSoup 结合的爬虫示例。

教学重点：不同解析方法适合不同网页结构。
"""

import requests
from bs4 import BeautifulSoup
import re

# --- 1. 请求网页 ---
url = "https://www.worldex.com.cn/"
headers = {"User-Agent": "Mozilla/5.0"}
resp = requests.get(url, headers=headers)
resp.encoding = resp.apparent_encoding   # 自动识别网页编码

print("状态码:", resp.status_code)
soup = BeautifulSoup(resp.text, "html.parser")  # 解析 HTML

# --- 2. 用正则表达式提取航线与报价 ---
print("\n=== 航线报价信息 ===")
# 正则模式：匹配"出发港+目的港+$价格/20GP$价格/40GP$价格/40HQ"格式
pattern = re.compile(r'([A-Z]+[A-Z])([A-Z\s,]+)\$(\d+)/20GP\$(\d+)/40GP\$(\d+)/40HQ')
for match in pattern.finditer(soup.get_text()):
    origin = match.group(1)         # 出发港
    dest = match.group(2).strip()   # 目的港
    price_20gp = match.group(3)     # 20GP 集装箱价格
    price_40gp = match.group(4)     # 40GP 集装箱价格
    price_40hq = match.group(5)     # 40HQ 集装箱价格
    print(f"{origin} → {dest}: 20GP={price_20gp}, 40GP={price_40gp}, 40HQ={price_40hq}")

# --- 3. 用关键词过滤提取新闻动态标题 ---
print("\n=== 新闻动态 ===")
for div in soup.find_all("div"):
    text = div.get_text(strip=True)
    if "新闻" in text or "动态" in text:   # 包含关键词的 div 才输出
        print(text)
