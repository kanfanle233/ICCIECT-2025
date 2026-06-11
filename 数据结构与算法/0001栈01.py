"""
栈（Stack）的基本操作示例。

教学重点：演示栈的后进先出（LIFO）特性，
包括入栈（push）、出栈（pop）、查看栈顶（peek）等操作。
使用 pythonds 库中的 Stack 类。
"""

# --- 1. 创建栈并入栈 ---
from pythonds.basic.stack import Stack

s = Stack()        # 创建一个空栈
s.push(4)          # 将整数 4 入栈
s.push('dog')      # 将字符串 'dog' 入栈

# --- 2. 查看栈顶元素（不弹出） ---
print(s.peek())    # 输出: dog（栈顶元素）

# --- 3. 继续入栈并查看栈信息 ---
s.push(True)       # 将布尔值 True 入栈
print(s.size())    # 输出: 3（当前栈中有 3 个元素）
print(s.isEmpty()) # 输出: False（栈不为空）

# --- 4. 出栈操作 ---
s.push(8.4)        # 再入栈一个浮点数
print(s.pop())     # 输出: 8.4（后进先出，最后入栈的先弹出）
print(s.pop())     # 输出: True
print(s.size())    # 输出: 2（弹出两个后剩余 2 个元素）