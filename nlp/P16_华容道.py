import tkinter as tk
from tkinter import messagebox
import random
from collections import deque
import warnings
warnings.filterwarnings("ignore")

class Klotski:
    def __init__(self, master):
        self.master = master
        self.master.title("经典华容道 - 横刀立马 (内置AI求解)")
        self.master.geometry("380x600")
        self.master.configure(bg='#1e272e')  # 极简现代深色风格
        self.master.resizable(False, False)

        self.cell_size = 80
        self.padding = 6
        self.moves = 0
        self.is_won = False
        self.is_playing_demo = False
        self.selected_block = None

        # 经典关卡："横刀立马"
        self.initial_layout = {
            '曹操': {'x': 1, 'y': 0, 'w': 2, 'h': 2, 'color': '#e74c3c', 'fg': '#ffffff'},
            '关羽': {'x': 1, 'y': 2, 'w': 2, 'h': 1, 'color': '#2ecc71', 'fg': '#ffffff'},
            '张飞': {'x': 0, 'y': 0, 'w': 1, 'h': 2, 'color': '#9b59b6', 'fg': '#ffffff'},
            '赵云': {'x': 3, 'y': 0, 'w': 1, 'h': 2, 'color': '#f1c40f', 'fg': '#2c3e50'},
            '马超': {'x': 0, 'y': 2, 'w': 1, 'h': 2, 'color': '#e67e22', 'fg': '#ffffff'},
            '黄忠': {'x': 3, 'y': 2, 'w': 1, 'h': 2, 'color': '#e84393', 'fg': '#ffffff'},
            '卒1': {'x': 1, 'y': 3, 'w': 1, 'h': 1, 'color': '#95a5a6', 'fg': '#2c3e50'},
            '卒2': {'x': 2, 'y': 3, 'w': 1, 'h': 1, 'color': '#95a5a6', 'fg': '#2c3e50'},
            '卒3': {'x': 1, 'y': 4, 'w': 1, 'h': 1, 'color': '#95a5a6', 'fg': '#2c3e50'},
            '卒4': {'x': 2, 'y': 4, 'w': 1, 'h': 1, 'color': '#95a5a6', 'fg': '#2c3e50'},
        }
        
        self.blocks = {}
        
        self.create_widgets()
        self.reset_game()

    def create_widgets(self):
        # 顶部栏
        self.header_frame = tk.Frame(self.master, bg='#1e272e')
        self.header_frame.pack(fill='x', padx=25, pady=(15, 10))
        
        self.title_label = tk.Label(
            self.header_frame, text="华容道", 
            font=('Helvetica', 22, 'bold'), fg='#ffffff', bg='#1e272e'
        )
        self.title_label.pack(side='left')
        
        self.moves_label = tk.Label(
            self.header_frame, text="步数: 0", 
            font=('Helvetica', 14, 'bold'), fg='#dcdde1', bg='#1e272e'
        )
        self.moves_label.pack(side='right', pady=5)

        # 棋盘容器
        canvas_width = 4 * self.cell_size
        canvas_height = 5 * self.cell_size
        self.border_frame = tk.Frame(self.master, bg='#2c3e50', bd=6, relief='ridge')
        self.border_frame.pack(padx=20, pady=10)
        
        self.canvas = tk.Canvas(
            self.border_frame, 
            width=canvas_width, 
            height=canvas_height, 
            bg='#2f3640', 
            highlightthickness=0
        )
        self.canvas.pack()

        # 绑定滑动事件
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # 底部双排按钮控制面板
        self.control_frame = tk.Frame(self.master, bg='#1e272e')
        self.control_frame.pack(fill='x', padx=25, pady=(10, 15))
        
        # 重新开始按钮
        self.reset_btn = tk.Label(
            self.control_frame, text="重置关卡", 
            font=('Helvetica', 12, 'bold'), fg='#ffffff', bg='#9c88ff',
            width=14, height=2, cursor='hand2', relief='flat'
        )
        self.reset_btn.pack(side='left', padx=5)
        self.reset_btn.bind("<Button-1>", lambda event: self.reset_game())
        self.reset_btn.bind("<Enter>", lambda event: self.reset_btn.config(bg='#786fa6') if not self.is_playing_demo else None)
        self.reset_btn.bind("<Leave>", lambda event: self.reset_btn.config(bg='#9c88ff') if not self.is_playing_demo else None)

        # AI 自动解题按钮
        self.ai_btn = tk.Label(
            self.control_frame, text="AI 自动演示", 
            font=('Helvetica', 12, 'bold'), fg='#ffffff', bg='#10ac84',
            width=14, height=2, cursor='hand2', relief='flat'
        )
        self.ai_btn.pack(side='right', padx=5)
        self.ai_btn.bind("<Button-1>", lambda event: self.start_ai_demo())
        self.ai_btn.bind("<Enter>", lambda event: self.ai_btn.config(bg='#0ca27a') if not self.is_playing_demo else None)
        self.ai_btn.bind("<Leave>", lambda event: self.ai_btn.config(bg='#10ac84') if not self.is_playing_demo else None)

    def reset_game(self):
        if self.is_playing_demo:
            self.is_playing_demo = False  # 中止 AI 演示
        self.moves = 0
        self.is_won = False
        self.selected_block = None
        self.moves_label.config(text="步数: 0")
        self.ai_btn.config(bg='#10ac84', text="AI 自动演示")
        
        self.blocks = {name: dict(attrs) for name, attrs in self.initial_layout.items()}
        self.draw_board()

    def draw_rounded_rect(self, x1, y1, x2, y2, radius=12, **kwargs):
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1
        ]
        return self.canvas.create_polygon(points, **kwargs, smooth=True)

    def draw_board(self):
        self.canvas.delete("all")
        
        # 网格引导线
        for i in range(1, 4):
            self.canvas.create_line(
                i * self.cell_size, 0, 
                i * self.cell_size, 5 * self.cell_size, 
                fill='#384250', dash=(2, 4), width=1
            )
        for j in range(1, 5):
            self.canvas.create_line(
                0, j * self.cell_size, 
                4 * self.cell_size, j * self.cell_size, 
                fill='#384250', dash=(2, 4), width=1
            )
            
        # 底部出口标志线
        self.canvas.create_rectangle(
            self.cell_size, 5 * self.cell_size - 4, 
            3 * self.cell_size, 5 * self.cell_size, 
            fill='#f1c40f', outline=''
        )

        for name, b in self.blocks.items():
            x1 = b['x'] * self.cell_size + self.padding
            y1 = b['y'] * self.cell_size + self.padding
            x2 = (b['x'] + b['w']) * self.cell_size - self.padding
            y2 = (b['y'] + b['h']) * self.cell_size - self.padding
            
            outline_color = '#ffffff' if name == self.selected_block else ''
            outline_width = 3 if name == self.selected_block else 0
            
            self.draw_rounded_rect(
                x1, y1, x2, y2, radius=12, 
                fill=b['color'], outline=outline_color, width=outline_width,
                tags=f"block_{name}"
            )
            
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            font_size = 20 if name == '曹操' else 16
            self.canvas.create_text(
                cx, cy, 
                text=name[:2] if len(name) > 2 else name, 
                font=('Helvetica', font_size, 'bold'), 
                fill=b['fg'], 
                tags=f"block_{name}"
            )

    def on_press(self, event):
        if self.is_won or self.is_playing_demo:
            return
        
        clicked_items = self.canvas.find_withtag("current")
        if not clicked_items:
            self.selected_block = None
            self.draw_board()
            return
            
        tags = self.canvas.gettags(clicked_items[0])
        for tag in tags:
            if tag.startswith("block_"):
                self.selected_block = tag.split("_")[1]
                break
                
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.draw_board()

    def on_release(self, event):
        if not self.selected_block or self.is_won or self.is_playing_demo:
            return
            
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        threshold = 25
        
        move_dir = None
        if abs(dx) > abs(dy) and abs(dx) > threshold:
            move_dir = (1, 0) if dx > 0 else (-1, 0)
        elif abs(dy) > abs(dx) and abs(dy) > threshold:
            move_dir = (0, 1) if dy > 0 else (0, -1)
            
        if move_dir:
            self.try_move(self.selected_block, move_dir[0], move_dir[1])
            
    def try_move(self, name, dx, dy):
        b = self.blocks[name]
        nx = b['x'] + dx
        ny = b['y'] + dy
        
        if nx < 0 or nx + b['w'] > 4 or ny < 0 or ny + b['h'] > 5:
            return False
            
        for other_name, ob in self.blocks.items():
            if other_name == name:
                continue
            if not (nx + b['w'] <= ob['x'] or ob['x'] + ob['w'] <= nx or
                    ny + b['h'] <= ob['y'] or ob['y'] + ob['h'] <= ny):
                return False
                
        b['x'] = nx
        b['y'] = ny
        self.moves += 1
        self.moves_label.config(text=f"步数: {self.moves}")
        
        self.draw_board()
        self.check_win()
        return True

    def check_win(self):
        caocao = self.blocks['曹操']
        if caocao['x'] == 1 and caocao['y'] == 3:
            self.is_won = True
            self.selected_block = None
            self.is_playing_demo = False
            self.draw_board()
            self.canvas.create_text(
                2 * self.cell_size, 4.5 * self.cell_size,
                text="逃离成功！", font=('Helvetica', 26, 'bold'), fill='#2ecc71'
            )
            messagebox.showinfo("大获全胜", f"解救成功！曹操已安全出逃！\n总步数: {self.moves}")

    # ==================== AI BFS 求解逻辑 ====================
    def solve_bfs(self):
        block_types = {
            '曹操': (2, 2), '关羽': (2, 1), '张飞': (1, 2), '赵云': (1, 2),
            '马超': (1, 2), '黄忠': (1, 2), '卒1': (1, 1), '卒2': (1, 1),
            '卒3': (1, 1), '卒4': (1, 1)
        }
        
        # 抓取当前滑块布局坐标
        start_state = {name: (b['x'], b['y']) for name, b in self.blocks.items()}
        
        def is_valid_state(pos):
            grid = [[False]*4 for _ in range(5)]
            for name, (x, y) in pos.items():
                w, h = block_types[name]
                if x < 0 or x + w > 4 or y < 0 or y + h > 5:
                    return False
                for r in range(y, y+h):
                    for c in range(x, x+w):
                        if grid[r][c]:
                            return False
                        grid[r][c] = True
            return True

        def get_all_moves(pos):
            possible_moves = []
            for name, (x, y) in pos.items():
                w, h = block_types[name]
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_pos = dict(pos)
                    new_pos[name] = (x + dx, y + dy)
                    if is_valid_state(new_pos):
                        possible_moves.append((name, dx, dy, new_pos))
            return possible_moves

        def state_key(pos):
            soldiers = sorted([pos['卒1'], pos['卒2'], pos['卒3'], pos['卒4']])
            verticals = sorted([pos['张飞'], pos['赵云'], pos['马超'], pos['黄忠']])
            return (pos['曹操'], pos['关羽'], tuple(verticals), tuple(soldiers))

        # BFS 队列
        queue = deque([(start_state, [])])
        visited = {state_key(start_state)}
        
        while queue:
            pos, path = queue.popleft()
            
            if pos['曹操'] == (1, 3):
                return path
                
            for name, dx, dy, next_pos in get_all_moves(pos):
                key = state_key(next_pos)
                if key not in visited:
                    visited.add(key)
                    queue.append((next_pos, path + [(name, dx, dy)]))
        return None

    def start_ai_demo(self):
        if self.is_won:
            return
            
        if self.is_playing_demo:
            # 如果已经在播放，再次点击代表“暂停/中止演示”
            self.is_playing_demo = False
            self.ai_btn.config(bg='#10ac84', text="AI 自动演示")
            return

        self.ai_btn.config(bg='#ea8214', text="中止演示")
        self.is_playing_demo = True
        
        # 寻找最短路径
        path = self.solve_bfs()
        if not path:
            messagebox.showinfo("AI 提示", "当前局面已无解，请尝试重置关卡！")
            self.is_playing_demo = False
            self.ai_btn.config(bg='#10ac84', text="AI 自动演示")
            return
            
        # 开始逐帧动画播放
        self.animate_path(path, 0)

    def animate_path(self, path, step_idx):
        if not self.is_playing_demo or self.is_won:
            return
            
        if step_idx >= len(path):
            self.is_playing_demo = False
            self.ai_btn.config(bg='#10ac84', text="AI 自动演示")
            return
            
        name, dx, dy = path[step_idx]
        self.try_move(name, dx, dy)
        
        # 250ms 走一步，行云流水的自动解密动画
        self.master.after(250, lambda: self.animate_path(path, step_idx + 1))

if __name__ == "__main__":
    root = tk.Tk()
    
    root.update_idletasks()
    width = 380
    height = 600
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    
    app = Klotski(root)
    root.mainloop()
