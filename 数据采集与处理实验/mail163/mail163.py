"""
使用 Selenium 自动登录 163 邮箱并爬取收件箱邮件列表。

教学重点：
- Selenium 处理 iframe 登录表单
- WebDriverWait 显式等待确保元素加载完成
- 处理动态渲染的邮件列表（aria-label 属性解析）
- 结果导出为 CSV 文件
"""

# --- 1. 导入模块 ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import csv
import time

# 登录账号（使用前需填写）
MAIL_USER = ""

# 自动下载并安装匹配的 ChromeDriver
service = Service(ChromeDriverManager().install())

# --- 2. aria-label 解析函数 ---
def parse_aria(label: str):
    """
    从 aria-label 里解析出 主题 / 发件人 / 时间。

    格式类似：
    "【重要】你有一个会员兑换码待领取 发件人： 网易邮箱助手 时间： 2025年4月14日 09:09（星期一）"
    """
    subject = label.strip()
    sender = ""
    dt = ""

    if " 发件人：" in label:
        subject, rest = label.split(" 发件人：", 1)
        subject = subject.strip()
        if " 时间：" in rest:
            sender, dt = rest.split(" 时间：", 1)
            sender = sender.strip()
            dt = dt.strip()

    return subject, sender, dt


# --- 3. 登录并爬取邮件主函数 ---
def login_and_scrape():
    """自动登录 163 邮箱，爬取收件箱邮件列表并导出为 CSV 文件。"""
    # --- 3.1 配置浏览器 ---
    options = Options()
    # 调试时先别无头，方便看
    # options.add_argument("--headless")

    driver = webdriver.Chrome(options=options, service=service)
    driver.maximize_window()

    # --- 3.2 打开登录页并切换到 iframe ---
    # 1. 打开登录页
    driver.get("https://mail.163.com/")

    # 2. 切到登录 iframe
    WebDriverWait(driver, 20).until(
        EC.frame_to_be_available_and_switch_to_it(
            (By.CSS_SELECTOR, 'iframe[id^="x-URS-iframe"]')
        )
    )

    # 3. 输入账号
    email_input = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'input[name="email"]'))
    )
    email_input.send_keys(MAIL_USER)

    # 4. 输入密码
    pwd = input("password: ")
    pwd_input = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "input.dlpwd"))
    )
    pwd_input.send_keys(pwd)

    # 5. 点击登录
    login_btn = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "#dologin"))
    )
    login_btn.click()

    # --- 3.3 登录后进入收件箱 ---
    # 回到顶层
    driver.switch_to.default_content()

    # 1）等待”收件箱”这个标签出现并可点击，然后点击它
    inbox_tab = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable(
            (By.XPATH, '//span[text()="收件箱"]')
        )
    )
    inbox_tab.click()

    # 2）等待收件箱邮件列表真正渲染出来
    list_div = WebDriverWait(driver, 30).until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, 'div[id$="_ListDiv"]')
        )
    )

    # 3）在列表里选出每一封邮件（role="link" & sign="letter"）
    mail_divs = list_div.find_elements(
        By.CSS_SELECTOR,
        'div[role="link"][sign="letter"]'
    )

    def parse_aria(label: str):
        # 解析 aria-label -> 主题 / 发件人 / 时间
        subject = label.strip()
        sender = ""
        dt = ""
        if " 发件人：" in label:
            subject, rest = label.split(" 发件人：", 1)
            subject = subject.strip()
            if " 时间：" in rest:
                sender, dt = rest.split(" 时间：", 1)
                sender = sender.strip()
                dt = dt.strip()
        return subject, sender, dt

    # --- 3.4 解析邮件信息 ---
    mails = []
    for div in mail_divs:
        label = div.get_attribute("aria-label") or ""
        if not label.strip():
            continue
        subject, sender, dt = parse_aria(label)
        mails.append(
            {
                "subject": subject,
                "sender": sender,
                "datetime": dt,
                "raw_aria": label,
            }
        )

    print(f"本页共抓到 {len(mails)} 封邮件")

    # --- 3.5 导出邮件列表为 CSV 文件 ---
    import csv
    with open("163_inbox_list.csv", "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig 保证 Excel 正确显示中文
        writer = csv.DictWriter(
            f, fieldnames=["subject", "sender", "datetime", "raw_aria"]
        )
        writer.writeheader()
        writer.writerows(mails)

    print("已保存到 163_inbox_list.csv")

    # 调试完你可以改成 driver.quit()
    # time.sleep(10)
    # driver.quit()


if __name__ == "__main__":
    login_and_scrape()
