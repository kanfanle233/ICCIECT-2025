"""
演示 requests Response 对象的常用属性和方法。

教学重点：
- requests.get() 使用 params 参数传递查询字符串
- Response 对象的核心属性：url、status_code、encoding、ok、cookies、headers
- apparent_encoding 与 encoding 的区别
- content（二进制）vs text（文本）的区别
"""

# --- 1. 导入模块 ---
import requests
import chardet

# --- 2. 发起带参数的 GET 请求 ---
res = requests.get(
    url='http://www.baidu.com/s',
    params={"wd": "立达学院"}  # 自动拼接到 URL 查询字符串中
)

# --- 3. 解码响应内容 ---
html = res.content.decode('utf-8', errors='ignore')              # 用 utf-8 解码，忽略错误字节
s = res.content.decode(chardet.detect(res.content)['encoding'])   # 用 chardet 检测编码再解码

# --- 4. 查看 Response 基本属性 ---
print('_' * 100)
print('url:', res.url, ',', type(res.url))               # 完整请求 URL（含参数）
print('status_code:', res.status_code, ',', type(res.status_code))  # HTTP 状态码
print('encoding:', res.encoding, ',', type(res.encoding))           # 响应编码
print('ok:', res.ok, ',', type(res.ok))                             # 状态码 < 400 则为 True

# --- 5. 查看 Cookies 和 Headers ---
print('_' * 50)
print("cookies:")
for ck in res.cookies:
    print(f"{ck.path}{ck.name}: {ck.value} {ck.expires}")  # 打印每个 cookie 的详细信息
print("headers:")
for h in res.headers:
    print(f"{h}: {res.headers[h]}")  # 打印所有响应头

# --- 6. apparent_encoding 检测 ---
print("_" * 50)
print('apparent_encoding:', res.apparent_encoding, type(res.apparent_encoding))
print(res.content.decode(res.apparent_encoding))  # 用 apparent_encoding 重新解码