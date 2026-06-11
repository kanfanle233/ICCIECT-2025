"""
演示 requests 请求的异常处理机制。

教学重点：
- requests 常见异常类型：ReadTimeout、ConnectionError、RequestException
- try-except 结构捕获不同类型的网络请求异常
- 异常应按从具体到笼统的顺序排列
"""

# --- 1. 导入模块 ---
import requests
from requests import ReadTimeout, ConnectionError, RequestException

# --- 2. 带异常处理的 HTTP 请求 ---
try:
    res = requests.get("http://www.baidu.com")
    print("status_code =", res.status_code)
except ReadTimeout:
    print("超时")               # 请求超时异常
except ConnectionError:
    print("连接错误")           # 网络连接失败
except RequestException:
    print("请求异常")           # requests 所有异常的基类，兜底捕获