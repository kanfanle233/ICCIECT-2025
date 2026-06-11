"""
学生成绩管理系统示例。

教学重点：用文本文件持久化学生编号、姓名和成绩。
"""

import os

# 文件名
filename = 'student.txt'

# 1. 输入并储存学生信息
def save_student(student_id, name, score):
    """将一条学生记录追加写入文件，'a' 模式表示追加而非覆盖。"""
    with open(filename, 'a') as file:
        file.write(f"{student_id},{name},{score}\n")

# 2. 打印所有学生信息
def print_students():
    """读取文件并逐行打印每位学生的学号、姓名和分数。"""
    if not os.path.exists(filename):
        print("没有学生信息。")
        return

    with open(filename, 'r') as file:
        for line in file:
            student_id, name, score = line.strip().split(',')
            print(f"学号: {student_id}, 姓名: {name}, 分数: {score}")

# 3. 修改学生信息
def modify_student(student_id, new_name, new_score):
    """根据学号找到对应记录并更新姓名和分数，写回文件。"""
    if not os.path.exists(filename):
        print("没有学生信息。")
        return

    updated = False
    lines = []
    with open(filename, 'r') as file:
        for line in file:
            id, name, score = line.strip().split(',')
            if id == student_id:
                lines.append(f"{student_id},{new_name},{new_score}\n")
                updated = True
            else:
                lines.append(line)

    if updated:
        with open(filename, 'w') as file:
            file.writelines(lines)
        print("学生信息已更新。")
    else:
        print("学生不存在。")

# 4. 删除学生信息
def delete_student(student_id):
    """根据学号删除对应记录，其余记录保留并写回文件。"""
    if not os.path.exists(filename):
        print("没有学生信息。")
        return

    found = False
    lines = []
    with open(filename, 'r') as file:
        for line in file:
            id, name, score = line.strip().split(',')
            if id == student_id:
                found = True
            else:
                lines.append(line)

    if found:
        with open(filename, 'w') as file:
            file.writelines(lines)
        print("学生信息已删除。")
    else:
        print("学生不存在。")

# 5. 按学生成绩排序
def sort_students():
    """读取所有学生信息，按分数从高到低排序后打印。"""
    if not os.path.exists(filename):
        print("没有学生信息。")
        return

    students = []
    with open(filename, 'r') as file:
        for line in file:
            student_id, name, score = line.strip().split(',')
            students.append((student_id, name, int(score)))

    students.sort(key=lambda x: x[2], reverse=True)

    for student in students:
        print(f"学号: {student[0]}, 姓名: {student[1]}, 分数: {student[2]}")

# 6. 查找学生信息
def find_student(student_id):
    """根据学号在文件中查找并打印对应学生的完整信息。"""
    if not os.path.exists(filename):
        print("没有学生信息。")
        return

    with open(filename, 'r') as file:
        for line in file:
            id, name, score = line.strip().split(',')
            if id == student_id:
                print(f"学号: {id}, 姓名: {name}, 分数: {score}")
                return

    print("学号不存在。")

# 主菜单
def main():
    while True:
        print("\n1. 输入并储存学生信息")
        print("2. 打印学生信息")
        print("3. 修改学生信息")
        print("4. 删除学生信息")
        print("5. 按学生成绩排序")
        print("6. 查找学生信息")
        print("7. 退出")
        choice = input("请选择功能: ")

        if choice == '1':
            student_id = input("输入学号: ")
            name = input("输入姓名: ")
            score = input("输入分数: ")
            save_student(student_id, name, score)
        elif choice == '2':
            print_students()
        elif choice == '3':
            student_id = input("输入要修改的学生学号: ")
            new_name = input("输入新的姓名: ")
            new_score = input("输入新的分数: ")
            modify_student(student_id, new_name, new_score)
        elif choice == '4':
            student_id = input("输入要删除的学生学号: ")
            delete_student(student_id)
        elif choice == '5':
            sort_students()
        elif choice == '6':
            student_id = input("输入要查找的学生学号: ")
            find_student(student_id)
        elif choice == '7':
            break
        else:
            print("无效的选择，请重新输入。")

if __name__ == "__main__":
    main()
