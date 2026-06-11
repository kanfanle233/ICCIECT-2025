"""
顺序表删除示例。

教学重点：删除元素前要检查位置是否合法，并把后面的元素向前移动。
"""

def ListDelete(L, i):
    """删除顺序表第 i 个位置的元素。"""
    # 顺序表的合法位置从 1 到 L.length，越界时不能删除。
    if i < 1 or i > L.length:
        return False
    for k in range(i, L.length):   # 从被删除位置的后一个元素开始向前搬。
        L.list[k-1] = L.list[k]    # 覆盖前一个位置，相当于填补空洞。
    L.length -= 1
    return True
