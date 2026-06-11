"""
爬取当当网图书搜索结果页面。

教学重点：
- requests 使用 params 参数构建搜索请求
- 处理网站编码不规范的问题（当当网 GB2312 编码）
- BeautifulSoup 解析并将结果保存为本地 HTML 文件
"""

# --- 1. 导入模块 ---
import requests
import chardet
import bs4

# --- 2. 当当网图书搜索函数 ---
def search_books(keyword):
    """在当当网搜索指定关键词的图书并保存结果页面。"""
    url = "http://search.dangdang.com"
    params = {
        "medium": "01",                          # 搜索范围参数
        "key": keyword,                          # 搜索关键词
        "category_path": "01.00.00.00.00.00"     # 图书分类路径
    }
    resp = requests.get(url, params=params)      # 发起带参数的 GET 请求

    if resp.status_code != 200:
        raise Exception(f"Found error (status_code={resp.status_code})")

    # 当当网的返回页面编码不规范，解码出错
    # content = resp.content.decode('GB2312')
    html = bs4.BeautifulSoup(resp.content, features="lxml").prettify()  # 直接用二进制内容解析
    print(html)

    # 将解析后的 HTML 保存到本地文件
    with open("temp.html","w")as f:
        f.write(html)

# --- 3. 主程序入口 ---
if __name__ == "__main__":
    search_books("大数据")  # 搜索关键词"大数据"