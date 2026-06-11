"""
正则表达式验证电话号码格式。

教学重点：使用 \\d{n} 量词精确匹配指定位数的数字。
"""

import re

def validate_phone_number(phone_number):
    """验证电话号码是否符合 xxx-xxxxxxxx 格式（3位区号-8位号码）。"""
    pattern = r'^\d{3}-\d{8}$'
    if re.match(pattern, phone_number):
        return True
    else:
        return False

phone_number = '123-45678901'
if validate_phone_number(phone_number):
    print("电话号码格式正确")
else:
    print("电话号码格式不正确")