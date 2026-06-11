"""
演示网页读取和 robots.txt 爬虫协议检查。

教学重点：
- requests 获取网页并处理非 200 状态码
- urllib.robotparser.RobotFileParser 解析 robots.txt
- 在爬虫前检查目标网站是否允许爬取
"""

# --- 1. 导入模块 ---
import requests
import chardet
from urllib.robotparser import RobotFileParser

# --- 2. 网页读取函数 ---
def read_uri(url):
    """获取指定 URL 的页面内容并自动解码。"""
    res = requests.get(url)
    if res.status_code != 200:
        raise Exception(f"Found error (status_code={res.status_code})")
    content = res.content.decode(chardet.detect(res.content)['encoding'])
    return content

# --- 3. robots.txt 检查函数 ---
def robots_allow(robots_url, useragent):
    """检查指定 User-Agent 是否被 robots.txt 允许爬取。"""
    rfp = RobotFileParser()
    rfp.set_url(robots_url)  # 设置 robots.txt 的 URL
    rfp.read()               # 读取并解析 robots.txt
    return rfp.can_fetch(useragent=useragent, url=robots_url)  # 判断是否允许爬取

# --- 4. 主程序入口 ---
if __name__ == '__main__':
    # print(read_uri("https://roll.news.sina.com.cn/news/gnxw/gdxw1/index.shtml"))
    print(robots_allow("http://roll.news.sina.com.cn/robots.txt",
                       "Mozilla/5.0 Firefox/57.0"))  # 检查 Firefox 是否被允许