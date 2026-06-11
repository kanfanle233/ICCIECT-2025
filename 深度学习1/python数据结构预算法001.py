"""
顺序表插入示例。

教学重点：插入元素前要检查位置是否合法，并把后面的元素向后移动。
"""

def ListInsert(L, i, e):
    """在链表第 i 个位置插入元素 e。"""
    j = 0
    p = L
    # p 从头结点开始向后走，直到找到第 i-1 个结点。
    while p != None and j < i-1:
        p = p.next
        j += 1
    if p == None or j > i-1:
        return False
    s = LNode(e)    # 新建结点，保存要插入的数据 e。
    s.next = p.next # 新结点先接住原来的后继，避免链表断开。
    p.next = s      # 前一个结点再指向新结点，插入完成。
    return True
