"""
使用 Selenium 动态渲染 + requests 下载百度图片搜索结果。

教学重点：
- Selenium 无头浏览器处理 JavaScript 动态渲染页面
- 多策略提取图片 URL（data 属性 / 背景图 / 详情页）
- requests 下载二进制文件并按扩展名保存
"""

# --- 1. 导入模块 ---
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import requests
import os
from pathlib import Path


def download_baidu_f1_images():
    """
    爬取百度图片搜索 F1 第一页所有图片（动态渲染版）。

    使用 Selenium 无头 Chrome 加载百度图片搜索页面，
    通过滚动触发懒加载，再提取图片 URL 并用 requests 下载到本地。
    """
    # --- 2. 配置 Chrome 无头模式 ---
    options = Options()
    options.add_argument('--headless')           # 无头模式，不显示浏览器窗口
    options.add_argument('--disable-gpu')        # 禁用 GPU 加速（无头模式兼容）
    options.add_argument('--no-sandbox')         # 禁用沙箱（Linux root 用户需要）
    options.add_argument('--window-size=1920,1080')  # 设置窗口大小
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

    driver = webdriver.Chrome(options=options)

    try:
        # --- 3. 访问百度图片搜索页面 ---
        url = "https://image.baidu.com/search/index?tn=baiduimage&word=f1"
        print("正在访问百度图片搜索...")
        driver.get(url)

        # **超长等待确保完全渲染**
        print("⏳ 等待页面渲染（10秒）...")
        time.sleep(10)

        # **深度滚动触发懒加载**
        print("📜 滚动页面加载图片...")
        for i in range(10):
            driver.execute_script(f"window.scrollTo(0, {i * 300});")
            time.sleep(1)

        # **动态获取所有<li>元素（图片容器）**
        print("\n🔍 正在查找图片容器...")
        list_items = driver.find_elements(By.CSS_SELECTOR, 'li[class*="waterfall-item"]')
        print(f"📦 找到 {len(list_items)} 个图片容器")

        if not list_items:
            print("❌ 未找到任何图片容器，请查看 debug.png")
            driver.save_screenshot('debug.png')  # 保存截图用于调试
            return

        # --- 5. 创建保存目录并开始下载 ---
        # **提取图片URL（从JavaScript变量或点击事件中）**
        save_dir = Path("f1_images")
        save_dir.mkdir(exist_ok=True)

        success_count = 0
        headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://image.baidu.com/'}  # 设置请求头，模拟浏览器访问

        for idx, item in enumerate(list_items[:20], 1):  # 限制前20张
            try:
                # **策略A：尝试从元素属性中获取（备用）**
                img_url = None

                # 方法1：直接查找<img>子元素（即使隐藏）
                imgs = item.find_elements(By.TAG_NAME, 'img')
                if imgs:
                    img = imgs[0]
                    img_url = img.get_attribute('data-imgurl') or img.get_attribute('src')

                # 方法2：用JavaScript获取背景图URL
                if not img_url:
                    bg_url = driver.execute_script("""
                        const style = window.getComputedStyle(arguments[0]);
                        const bg = style.backgroundImage;
                        return bg.match(/url\\(['"]?([^'"]+)['"]?\\)/)?.[1] || null;
                    """, item)
                    img_url = bg_url

                # 方法3：从<a>标签的href中提取（需访问详情页）
                if not img_url:
                    detail_link = item.find_element(By.TAG_NAME, 'a').get_attribute('href')
                    print(f"⚠️ 第{idx}张需从详情页获取: {detail_link}")
                    continue  # 跳过详情页逻辑（较复杂）

                if not img_url or img_url.startswith('data:'):
                    print(f"⚠️ 第{idx}张未找到有效URL，跳过")
                    continue

                print(f"📥 正在下载第 {idx} 张: {img_url[:60]}...")

                # **下载图片**
                response = requests.get(img_url, headers=headers, timeout=15)
                if response.status_code == 200:
                    # 确定扩展名
                    file_extension = 'jpg'
                    if '.' in img_url:
                        ext = img_url.split('.')[-1].split('?')[0].lower()
                        if ext in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                            file_extension = ext

                    save_path = save_dir / f"f1_{idx:02d}.{file_extension}"
                    with open(save_path, 'wb') as f:
                        f.write(response.content)

                    success_count += 1
                    print(f"✅ 已保存: {save_path.name} ({len(response.content)} bytes)")
                else:
                    print(f"❌ 下载失败，状态码: {response.status_code}")

            except Exception as e:
                print(f"❌ 第{idx}张处理失败: {type(e).__name__}: {e}")
                continue

        print(f"\n🎉 下载完成！成功 {success_count}/{len(list_items)} 张")
        print(f"📁 图片保存在: {os.path.abspath(save_dir)}")

    except Exception as e:
        print(f"❌ 爬取失败: {type(e).__name__}: {e}")
        driver.save_screenshot('debug.png')  # 异常时保存截图便于排查
        print("📸 已保存调试截图: debug.png")

    finally:
        driver.quit()  # 无论成功或失败，都关闭浏览器释放资源
        print("🚪 浏览器已关闭")


if __name__ == '__main__':
    download_baidu_f1_images()