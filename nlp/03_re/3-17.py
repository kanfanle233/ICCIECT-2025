"""
正则表达式验证邮箱地址格式。

教学重点：使用 re.match 结合正则表达式
匹配符合 xxx@xxx.xxx 格式的邮箱地址。
"""

import re

def match_email_address(email):
    """验证邮箱地址格式是否合法。"""
    pattern = r'^[\w\-.]+@[\w\-.]+\.\w+$'  # 用户名@域名.后缀
    if re.match(pattern, email):
        return True
    else:
        return False

email1 = "test@example.com"
email2 = "invalid_email"
print(match_email_address(email1))  # 输出 True
print(match_email_address(email2))  # 输出 False