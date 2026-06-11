import tkinter as tk
from tkinter import messagebox
import random

class EightPuzzle:
    def __init__(self, master):
        self.master = master
        self.master.title("八数码经典挑战")
        self.master.geometry("360x460")
        self.master.configure(bg='#1e272e')  # 深色背景极简现代风
        self.master.resizable(False, False)

        self.state = []
        self.empty_pos = (2, 2)  # row, col
        self.labels = {}
        self.moves = 0
        self.is_won = False
        
        self.create_widgets()
        self.new_game()

    def create_widgets(self):
        # 顶部栏：标题 + 计步器
        self.header_frame = tk.Frame(self.master, bg='#1e272e')
        self.header_frame.pack(fill='x', padx=20, pady=(20, 10))
        
        self.title_label = tk.Label(
            self.header_frame, text="8-PUZZLE", 
            font=('Helvetica', 22, 'bold'), fg='#f5f6fa', bg='#1e272e'
        )
        self.title_label.pack(side='left')
        
        self.moves_label = tk.Label(
            self.header_frame, text="步数: 0", 
            font=('Helvetica', 14, 'bold'), fg='#dcdde1', bg='#1e272e'
        )
        self.moves_label.pack(side='right', pady=5)

        # 游戏区域面板（使用深灰色卡槽背景）
        self.grid_container = tk.Frame(self.master, bg='#2f3640', padx=8, pady=8)
        self.grid_container.pack(padx=20, pady=10)
        
        # 使用 Label 自定义数字卡片，彻底规避 macOS 上原生 Button 无法更改背景色的硬伤
        for r in range(3):
            for c in range(3):
                lbl = tk.Label(
                    self.grid_container, text="", 
                    font=('Helvetica', 26, 'bold'),
                    width=5, height=2,
                    bd=0, relief='flat'
                )
                lbl.grid(row=r, column=c, padx=4, pady=4)
                
                # 绑定事件：点击、鼠标进入悬停、鼠标离开
                lbl.bind("<Button-1>", lambda event, row=r, col=c: self.on_tile_click(row, col))
                lbl.bind("<Enter>", lambda event, row=r, col=c: self.on_hover_enter(row, col))
                lbl.bind("<Leave>", lambda event, row=r, col=c: self.on_hover_leave(row, col))
                self.labels[(r, c)] = lbl
                
        # 底部控制区
        self.control_frame = tk.Frame(self.master, bg='#1e272e')
        self.control_frame.pack(fill='x', pady=(15, 20))
        
        # 使用 Label 优雅定制的扁平化重开按钮
        self.reset_btn = tk.Label(
            self.control_frame, text="重新打乱", 
            font=('Helvetica', 13, 'bold'), fg='#ffffff', bg='#9c88ff',
            width=15, height=2, cursor='hand2', relief='flat'
        )
        self.reset_btn.pack(anchor='center')
        self.reset_btn.bind("<Button-1>", lambda event: self.new_game())
        self.reset_btn.bind("<Enter>", lambda event: self.reset_btn.config(bg='#786fa6'))
        self.reset_btn.bind("<Leave>", lambda event: self.reset_btn.config(bg='#9c88ff'))

    def new_game(self):
        self.moves = 0
        self.is_won = False
        self.moves_label.config(text="步数: 0")
        
        # 初始目标状态
        state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        self.empty_pos = (2, 2)
        
        # 随机移动打乱拼图，保证 100% 有解
        for _ in range(120):
            r, c = self.empty_pos
            moves = []
            if r > 0: moves.append((r-1, c))
            if r < 2: moves.append((r+1, c))
            if c > 0: moves.append((r, c-1))
            if c < 2: moves.append((r, c+1))
            
            nr, nc = random.choice(moves)
            state[r][c], state[nr][nc] = state[nr][nc], state[r][c]
            self.empty_pos = (nr, nc)
            
        self.state = state
        self.update_ui()

    def on_tile_click(self, row, col):
        if self.is_won:
            return
            
        er, ec = self.empty_pos
        # 检查点击的方块是否与空格相邻
        if (abs(row - er) == 1 and col == ec) or (abs(col - ec) == 1 and row == er):
            # 交换数据
            self.state[er][ec], self.state[row][col] = self.state[row][col], self.state[er][ec]
            self.empty_pos = (row, col)
            self.moves += 1
            self.moves_label.config(text=f"步数: {self.moves}")
            self.update_ui()
            
            # 手动更新一次悬停状态以更新指针类型和背景颜色
            self.on_hover_enter(er, ec)
            
            self.check_win()

    def on_hover_enter(self, row, col):
        if self.is_won:
            return
        
        er, ec = self.empty_pos
        # 判断该滑块是否可被移动（是否与空格相邻）
        is_adjacent = (abs(row - er) == 1 and col == ec) or (abs(col - ec) == 1 and row == er)
        
        if self.state[row][col] != 0:
            if is_adjacent:
                # 可移动的方块：悬停变鲜艳亮蓝，指针变成小手手
                self.labels[(row, col)].config(bg='#00a8ff', cursor='hand2')
            else:
                # 不可移动的方块：光标提示不可用
                self.labels[(row, col)].config(cursor='no')

    def on_hover_leave(self, row, col):
        if self.is_won:
            return
        if self.state[row][col] != 0:
            # 恢复静止状态的高级深蓝色
            self.labels[(row, col)].config(bg='#0984e3')

    def update_ui(self):
        for r in range(3):
            for c in range(3):
                val = self.state[r][c]
                lbl = self.labels[(r, c)]
                if val == 0:
                    # 空位：与灰色卡槽完全融为一体，形成空槽视觉
                    lbl.config(text="", bg='#2f3640')
                else:
                    # 数字卡片：高保真扁平蓝白配
                    lbl.config(text=str(val), fg='#ffffff', bg='#0984e3')

    def check_win(self):
        target = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        if self.state == target:
            self.is_won = True
            # 胜利彩蛋：所有数字块渐变为高亮翡翠绿
            for r in range(3):
                for c in range(3):
                    if self.state[r][c] != 0:
                        self.labels[(r, c)].config(bg='#4cd137', fg='#ffffff')
            messagebox.showinfo("胜利", f"太牛了！你成功复原了拼图！\n共计步数: {self.moves}")

if __name__ == "__main__":
    root = tk.Tk()
    
    # 居中屏幕
    root.update_idletasks()
    width = 360
    height = 460
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    
    app = EightPuzzle(root)
    root.mainloop()
