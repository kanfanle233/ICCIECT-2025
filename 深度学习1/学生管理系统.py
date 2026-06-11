"""
学生信息管理系统示例。

教学重点：用列表/文件保存结构化数据，并提供增删改查操作。
"""

# 用一个列表来存储所有学生的信息
students = []  #定义一个全局变量 students，用于存储所有学生的信息。这个变量是一个列表，每个学生的信息以字典形式存储在列表中。


# 1. 输入并储存学生信息
def input_student():
    student_id = input("输入学号: ")
    name = input("输入姓名: ")
    score = input("输入分数: ")  # 分数保持为字符串类型，避免初学者处理数据类型转换
    student = {'id': student_id, 'name': name, 'score': score}  # 创建一个学生字典
    students.append(student)  # 将学生信息添加到列表中
    print("学生信息已保存。")


# 2. 打印所有学生信息
def print_students():
    if len(students) == 0:  # 检查列表是否为空
        print("没有学生信息。")
    else:
        for student in students:  # 遍历列表中的每个学生
            print(f"学号: {student['id']}, 姓名: {student['name']}, 分数: {student['score']}")


# 3. 修改学生信息
def modify_student():
    student_id = input("输入要修改的学生学号: ")
    for student in students:  # 遍历列表中的每个学生
        if student['id'] == student_id:  # 找到匹配的学号
            student['name'] = input("输入新的姓名: ")
            student['score'] = input("输入新的分数: ")
            print("学生信息已更新。")
            return
    print("学生不存在。")


# 4. 删除学生信息
def delete_student():
    student_id = input("输入要删除的学生学号: ")
    for student in students:  # 遍历列表中的每个学生
        if student['id'] == student_id:  # 找到匹配的学号
            students.remove(student)  # 从列表中移除学生
            print("学生信息已删除。")
            return
    print("学生不存在。")


# 5. 按学生成绩排序
def sort_students():
    if len(students) == 0:
        print("没有学生信息。")
    else:
        students.sort(key=lambda x: int(x['score']), reverse=True)

    print("按成绩排序后的学生信息：")
    for student in students:
        score = int(student['score'])
        if score < 60:
            grade = "不及格"
        elif score < 80:
            grade = "及格"
        else:
            grade = "优秀"
        print(f"学号: {student['id']}, 姓名: {student['name']}, 分数: {student['score']}, 评价: {grade}")


# 6. 查找学生信息
def find_student():
    student_id = input("输入要查找的学生学号: ")
    for student in students:  # 遍历列表中的每个学生
        if student['id'] == student_id:  # 找到匹配的学号
            print(f"学号: {student['id']}, 姓名: {student['name']}, 分数: {student['score']}")
            return
    print("学号不存在。")

# 7. 从文件导入学生信息
def import_students():
    try:
        with open("students.txt", "r") as file:
            for line in file:
                student_id, name, score = line.strip().split(',')
                students.append({'id': student_id, 'name': name, 'score': score})
        print("学生信息已从 students.txt 文件导入。") #fxz
    except FileNotFoundError:
        print("文件不存在。")


# 8. 导出学生信息到文件
def export_students():
    with open("students.txt", "w") as file:
        for student in students:
            file.write(f"{student['id']},{student['name']},{student['score']}\n")
    print("学生信息已导出到 students.txt 文件。")



# 主菜单
def main():
    while True:
        print("\n1. 输入并储存学生信息")
        print("2. 打印学生信息")
        print("3. 修改学生信息")
        print("4. 删除学生信息")
        print("5. 按学生成绩排序")
        print("6. 查找学生信息")
        print("7. 导出学生信息到文件")
        print("8. 从文件导入学生信息")
        print("9. 退出")

        choice = input("请选择功能: ")

        if choice == '1':
            input_student()  # 调用输入学生信息的函数
        elif choice == '2':
            print_students()  # 调用打印学生信息的函数
        elif choice == '3':
            modify_student()  # 调用修改学生信息的函数
        elif choice == '4':
            delete_student()  # 调用删除学生信息的函数
        elif choice == '5':
            sort_students()  # 调用按成绩排序的函数
        elif choice == '6':
            find_student()  # 调用查找学生信息的函数
        elif choice == '7':
            export_students()
        elif choice == '8':
            import_students()
        elif choice == '9':
            break # 退出循环，结束程序
        else:
            print("无效的选择，请重新输入。")


if __name__ == "__main__":
    main()  # 运行主菜单fxz

# #查当前脚本是否作为主程序运行。
#如果当前脚本作为主程序运行，则调用 main() 函数，启动学生信息管理系统。