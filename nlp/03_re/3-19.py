"""
正则表达式验证日期格式。

教学重点：使用 \\d{4}-\\d{2}-\\d{2} 匹配 YYYY-MM-DD 格式的日期。
"""

import re
def validate_date(date):
    """验证日期是否符合 YYYY-MM-DD 格式。"""
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    if re.match(pattern, date):
        return True
    else:
        return False

date = '2023-10-27'
if validate_date(date):
    print("日期格式正确")
else:
    print("日期格式不正确")