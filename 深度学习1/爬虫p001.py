"""
requests 和 BeautifulSoup 爬虫入门。

教学重点：先请求网页，再解析 HTML。
"""

import requests
from bs4 import BeautifulSoup

# --- 1. 请求设置 ---
base_url = "https://movie.douban.com/top250"
headers = {
    "User-Agent": "Mozilla/5.0",
    "Cookie": "..."  # 实际运行时需要替换为有效的 Cookie
}

# --- 2. 分页爬取 Top 250 ---
pages = 10  # 豆瓣 Top 250 共 10 页，每页 25 部电影
for page in range(pages):
    # start 参数控制从第几部电影开始，实现翻页效果
    myparams = {"start": page * 25, "filter": ""}
    response = requests.get(base_url, headers=headers, params=myparams)

    if response.status_code != 200:
        print(f"第 {page + 1} 页失败")
        continue

    # --- 3. 解析 HTML，提取电影信息 ---
    soup = BeautifulSoup(response.text, "html.parser")
    for li in soup.select("ol.grid_view li"):         # 每个 li 标签对应一部电影
        title = li.find("span", class_="title").text   # 电影标题
        rating = li.find("span", class_="rating_num").text  # 评分
        count = li.find_all("span")[-2].text            # 评价人数
        print(f"{title}  评分：{rating}  评价：{count}")