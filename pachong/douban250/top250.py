"""
豆瓣电影Top250爬虫 —— 使用 Selenium 无头浏览器方案。

教学重点：
- Selenium WebDriver 的初始化与无头模式配置
- WebDriverWait 显式等待，确保页面元素加载完毕
- CSS 选择器定位复杂嵌套元素（电影卡片）
- 正则表达式提取评论数
- 数据格式化输出到 txt 文件
"""

# --- 1. 导入模块 ---
import time              # 控制爬取间隔，避免请求过快
import random            # 生成随机延迟时间
import re                # 正则表达式，用于提取评论数中的数字
from datetime import datetime  # 获取当前时间戳用于记录爬取时间
from selenium import webdriver  # Selenium 核心，驱动浏览器
from selenium.webdriver.common.by import By  # 定位方式（CLASS_NAME, CSS_SELECTOR 等）
from selenium.webdriver.support.ui import WebDriverWait  # 显式等待器
from selenium.webdriver.support import expected_conditions as EC  # 等待条件（如元素出现）
from selenium.webdriver.chrome.options import Options  # Chrome 浏览器选项配置


# --- 2. 浏览器初始化 ---

def init_driver():
    """初始化 Selenium WebDriver（无头模式）"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')          # 无头模式，不弹出浏览器窗口
    chrome_options.add_argument('--disable-gpu')       # 禁用 GPU 加速，兼容无显卡环境
    chrome_options.add_argument('--no-sandbox')        # 禁用沙箱，Linux / Docker 环境必需
    chrome_options.add_argument('--disable-dev-shm-usage')  # 解决 Docker 中 /dev/shm 空间不足的问题
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')  # 移除自动化检测标志，降低被反爬拦截的概率
    chrome_options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')  # 伪装为正常 Chrome 浏览器

    driver = webdriver.Chrome(options=chrome_options)  # 创建 Chrome WebDriver 实例
    driver.implicitly_wait(10)  # 隐式等待 10 秒，查找元素时自动重试
    return driver


# --- 3. 数据提取 ---

def extract_movie_data(driver):
    """从当前页面提取电影数据"""
    movies = []
    wait = WebDriverWait(driver, 10)  # 创建显式等待器，最多等待 10 秒
    wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "item")))  # 等待 class="item" 的元素全部加载完毕

    movie_items = driver.find_elements(By.CLASS_NAME, "item")

    for item in movie_items:
        try:
            # 提取电影名称（中文+外文）
            title_elements = item.find_elements(By.CSS_SELECTOR, ".info .hd .title")
            titles = [t.text.strip() for t in title_elements]
            full_title = " / ".join(titles)

            # 提取评分
            rating = item.find_element(By.CLASS_NAME, "rating_num").text.strip()

            # 提取评论数
            comment_info = item.find_element(By.CSS_SELECTOR, ".star span:last-child").text
            comment_match = re.search(r'(\d+)', comment_info)
            comment_count = comment_match.group(1) if comment_match else "0"

            movies.append({
                "电影名称": full_title,
                "评分": rating,
                "评论数": comment_count
            })

        except Exception:
            continue

    return movies


# --- 4. 数据保存 ---

def save_to_txt(movies, filename="douban_top250_movies.txt"):
    """保存数据到 txt 文件"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("豆瓣电影Top250数据\n")
        f.write(f"爬取时间: {timestamp}\n")
        f.write("=" * 60 + "\n\n")

        # 写入表头
        f.write(f"{'排名':<6} {'电影名称':<50} {'评分':<6} {'评论数':<10}\n")
        f.write("-" * 80 + "\n")

        # 写入数据
        for i, movie in enumerate(movies, 1):
            # 处理电影名称过长的情况
            title = movie['电影名称'][:48] + "..." if len(movie['电影名称']) > 50 else movie['电影名称']
            f.write(f"{i:<6} {title:<50} {movie['评分']:<6} {movie['评论数']:<10}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write(f"总计: {len(movies)} 部电影\n")
        f.write("=" * 60)


# --- 5. 主爬取流程 ---

def scrape_douban_top250():
    """主函数：爬取豆瓣 Top250 电影"""
    base_url = "https://movie.douban.com/top250"  # 豆瓣 Top250 基础 URL
    all_movies = []  # 汇总所有页面的电影数据

    driver = init_driver()  # 初始化无头浏览器

    try:
        for page in range(0, 250, 25):  # 豆瓣 Top250 共 10 页，每页 25 条，start 参数从 0 到 225
            url = f"{base_url}?start={page}&filter="  # 拼接分页 URL，start 表示偏移量
            print(f"正在爬取: {url}")

            driver.get(url)
            time.sleep(random.uniform(2, 4))

            page_movies = extract_movie_data(driver)
            all_movies.extend(page_movies)

            print(f"已爬取 {len(all_movies)} 部电影")

        # 保存到txt文件
        save_to_txt(all_movies)
        print(f"\n爬取完成！共 {len(all_movies)} 部电影数据已保存到 douban_top250_movies.txt")

    except Exception as e:
        print(f"爬取过程中出错: {e}")
    finally:
        driver.quit()

    return all_movies


# --- 6. 程序入口 ---

if __name__ == "__main__":
    movies_data = scrape_douban_top250()  # 调用主爬取函数

    # 显示前5条数据预览
    print("\n前5条数据预览:")
    for i, movie in enumerate(movies_data[:5], 1):
        print(f"{i}. {movie['电影名称']} | 评分: {movie['评分']} | 评论数: {movie['评论数']}")