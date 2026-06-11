"""
使用 Selenium 自动化爬取优志愿网站 985 高校列表数据。

教学重点：
- Selenium 处理 iframe 嵌套页面
- 模拟点击筛选条件（985 标签）
- 滚动加载更多内容并提取结构化数据
- webdriver_manager 自动管理 ChromeDriver
"""

# --- 1. 环境配置与模块导入 ---
import os
# 在导入webdriver_manager之前添加
os.environ['WDM_SSL_VERIFY'] = '0'  # 禁用 SSL 验证，避免证书问题
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time

# 自动下载并安装匹配的 ChromeDriver
service = Service(ChromeDriverManager().install())

# --- 2. 爬虫主函数 ---
def spide():
    """爬取优志愿网站的 985 高校列表，打印高校名称、排名和热度。"""
    # --- 2.1 配置浏览器并访问目标页面 ---
    options = Options()
    options.headless = True  # 无头模式

    driver = webdriver.Chrome(options=options, service=service)
    driver.maximize_window()  # 最大化窗口，确保元素可见
    url = "https://www.youzy.cn/tzy/search/colleges/collegeList"
    driver.get(url)
    time.sleep(1)
    # 首次访问上述网址没有成功，须进行第二次访问
    driver.get(url)

    # 如果数据在iframe中，则要执行下面两句
    iframe = driver.find_element(By.CSS_SELECTOR, '#youzy_part_view')
    driver.switch_to.frame(iframe)

    # 点击985
    element = driver.find_element(by=By.XPATH, value='//span[contains(text(), "985")]')
    element.click()     # element.send_keys('abc')
    # input()

    # 列出所有985高校
    for _ in range(3):  # 滚动条移到最下方以便加载更多大学，移动3次
        element = driver.find_element(by=By.CSS_SELECTOR, value='.all-colleges')
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight;", element)
        time.sleep(2)

    # --- 2.4 提取并打印 985 高校信息 ---
    num = 0
    for univ in driver.find_elements(by=By.CSS_SELECTOR, value='.college-list.mb30'):
        name = univ.find_element(by=By.CSS_SELECTOR, value='a.f20')  # 大学名称
        order, heat = univ.find_elements(by=By.CSS_SELECTOR, value='.heat.f12 span')  # 排名和热度
        num += 1
        name = name.text.replace("\n", "")     # 清理文本中的换行符
        order = order.text.replace("\n", "")
        heat = heat.text.replace("\n", "")
        print(f'{num}. 大学: {name},\t排名: {order},\t热度: {heat}')

if __name__ == '__main__':
    spide()
