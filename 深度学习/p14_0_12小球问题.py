#!/usr/bin/env python3
"""
12小球问题求解演示
问题描述：有12个外表看起来一样的小球，11个是好的，1个是次品
好的重量一样，不好的重量不一样（不知道更重还是更轻）
使用天平秤3次可以识别出坏的球
"""

import tkinter as tk
import time
import threading
import random

class TwelveBallPuzzle:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("12小球问题求解演示")
        self.root.geometry("1000x700")

        # 小球状态：0=正常，1=较重，-1=较轻
        self.balls = [0] * 12
        self.bad_ball = random.randint(0, 11)
        self.bad_ball_type = random.choice([1, -1])  # 1=重，-1=轻
        self.balls[self.bad_ball] = self.bad_ball_type

        self.setup_ui()
        self.animation_running = False

    def setup_ui(self):
        """设置UI界面"""
        title = tk.Label(self.root, text="12小球问题求解演示", font=("Arial", 20, "bold"))
        title.pack(pady=10)

        desc = tk.Label(self.root, text="12个小球中有1个重量不同（较重或较轻），用天平秤3次找出",
                       font=("Arial", 12))
        desc.pack()

        self.canvas = tk.Canvas(self.root, width=900, height=400, bg="white")
        self.canvas.pack(pady=20)

        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        self.start_btn = tk.Button(control_frame, text="开始演示", command=self.start_demo,
                                   font=("Arial", 12), bg="green", fg="white")
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = tk.Button(control_frame, text="重置", command=self.reset,
                                   font=("Arial", 12))
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        self.info_label = tk.Label(self.root, text="", font=("Arial", 11), wraplength=800)
        self.info_label.pack(pady=10)

        self.result_label = tk.Label(self.root, text="", font=("Arial", 12, "bold"))
        self.result_label.pack(pady=5)

        self.draw_balls()
        self.draw_scale()

    def draw_balls(self):
        """绘制小球"""
        self.canvas.delete("balls")
        for i in range(12):
            x = 50 + (i % 6) * 140
            y = 50 if i < 6 else 150
            self.canvas.create_oval(x-25, y-25, x+25, y+25,
                                   fill="#4CAF50", outline="black", width=2, tags="balls")
            self.canvas.create_text(x, y, text=str(i+1), font=("Arial", 12, "bold"), tags="balls")

    def draw_scale(self):
        """绘制天平"""
        self.canvas.delete("scale")
        self.canvas.create_rectangle(420, 300, 480, 320, fill="brown", tags="scale")
        self.canvas.create_rectangle(445, 250, 455, 300, fill="brown", tags="scale")
        self.canvas.create_polygon(300, 250, 600, 250, 550, 280, 350, 280,
                                   fill="lightgray", outline="black", tags="scale")
        self.canvas.create_polygon(600, 250, 900, 250, 850, 280, 650, 280,
                                   fill="lightgray", outline="black", tags="scale")
        self.canvas.create_text(450, 265, text="左盘", font=("Arial", 10), tags="scale")
        self.canvas.create_text(750, 265, text="右盘", font=("Arial", 10), tags="scale")

    def show_result(self, result, left_balls, right_balls):
        """显示称重结果"""
        self.canvas.delete("result")
        for i, ball_id in enumerate(left_balls):
            x = 350 + i * 60
            self.canvas.create_oval(x-15, 230, x+15, 260, fill="orange", outline="black", tags="result")
            self.canvas.create_text(x, 245, text=str(ball_id), font=("Arial", 9), tags="result")

        for i, ball_id in enumerate(right_balls):
            x = 650 + i * 60
            self.canvas.create_oval(x-15, 230, x+15, 260, fill="blue", outline="black", tags="result")
            self.canvas.create_text(x, 245, text=str(ball_id), font=("Arial", 9), tags="result")

        if result == "left_heavy":
            self.canvas.create_text(450, 310, text="▼", font=("Arial", 16), fill="red", tags="result")
            self.canvas.create_text(750, 310, text="▲", font=("Arial", 16), fill="green", tags="result")
        elif result == "right_heavy":
            self.canvas.create_text(450, 310, text="▲", font=("Arial", 16), fill="green", tags="result")
            self.canvas.create_text(750, 310, text="▼", font=("Arial", 16), fill="red", tags="result")
        else:
            self.canvas.create_text(600, 310, text="=", font=("Arial", 16), fill="blue", tags="result")

    def weigh(self, left_balls, right_balls):
        """模拟称重"""
        left_weight = sum(self.balls[b-1] for b in left_balls)
        right_weight = sum(self.balls[b-1] for b in right_balls)
        if left_weight > right_weight:
            return "left_heavy"
        elif left_weight < right_weight:
            return "right_heavy"
        else:
            return "balanced"

    def highlight_balls(self, ball_ids, color):
        """高亮显示小球"""
        self.canvas.delete("highlight")
        for ball_id in ball_ids:
            i = ball_id - 1
            x = 50 + (i % 6) * 140
            y = 50 if i < 6 else 150
            self.canvas.create_oval(x-28, y-28, x+28, y+28, outline=color, width=3, tags="highlight")

    def mark_ball(self, ball_id, color, text=""):
        """标记小球"""
        i = ball_id - 1
        x = 50 + (i % 6) * 140
        y = 50 if i < 6 else 150
        self.canvas.create_oval(x-25, y-25, x+25, y+25, fill=color, outline="black", width=2, tags="result")
        self.canvas.create_text(x, y, text=str(i+1), font=("Arial", 12, "bold"), tags="result")
        if text:
            self.canvas.create_text(x, y+35, text=text, font=("Arial", 9, "bold"), fill="red", tags="result")

    def animate_weighing(self, left_balls, right_balls, result):
        """动画演示称重过程"""
        self.highlight_balls(left_balls, "orange")
        self.highlight_balls(right_balls, "blue")
        self.root.update()
        time.sleep(0.5)
        self.show_result(result, left_balls, right_balls)
        self.root.update()
        time.sleep(1)

    def start_demo(self):
        """开始演示"""
        if self.animation_running:
            return
        self.animation_running = True
        self.start_btn.config(state=tk.DISABLED)
        thread = threading.Thread(target=self.run_demo)
        thread.daemon = True
        thread.start()

    def run_demo(self):
        """运行演示 - 使用正确的12小球算法"""
        steps = 0

        # === 第一次称重：1-4 vs 5-8 ===
        self.info_label.config(text="第一次称重：比较球 1-4 和 球 5-8")
        left_balls = [1, 2, 3, 4]
        right_balls = [5, 6, 7, 8]
        result1 = self.weigh(left_balls, right_balls)
        steps += 1
        self.animate_weighing(left_balls, right_balls, result1)

        if result1 == "balanced":
            # 坏球在9-12中
            self.info_label.config(text="第一次结果：平衡 → 坏球在球 9-12 中")
            self.root.update()
            time.sleep(1)

            # 第二次称重：9-11 vs 1-3（已知正常）
            self.info_label.config(text="第二次称重：比较球 9,10,11 和 球 1,2,3（已知正常）")
            left_balls = [9, 10, 11]
            right_balls = [1, 2, 3]
            result2 = self.weigh(left_balls, right_balls)
            steps += 1
            self.animate_weighing(left_balls, right_balls, result2)

            if result2 == "balanced":
                self.info_label.config(text="第二次结果：平衡 → 坏球是球 12")
                self.root.update()
                time.sleep(1)

                # 第三次：12 vs 1
                self.info_label.config(text="第三次称重：比较球 12 和 球 1（已知正常）")
                left_balls = [12]
                right_balls = [1]
                result3 = self.weigh(left_balls, right_balls)
                steps += 1
                self.animate_weighing(left_balls, right_balls, result3)

                if result3 == "left_heavy":
                    self.result_label.config(text="结果：球 12 较重", fg="red")
                    self.mark_ball(12, "red", "较重")
                else:
                    self.result_label.config(text="结果：球 12 较轻", fg="blue")
                    self.mark_ball(12, "lightblue", "较轻")

            elif result2 == "left_heavy":
                self.info_label.config(text="第二次结果：左盘重 → 坏球在 9-11 中且较重")
                self.root.update()
                time.sleep(1)

                # 第三次：9 vs 10
                self.info_label.config(text="第三次称重：比较球 9 和 球 10")
                left_balls = [9]
                right_balls = [10]
                result3 = self.weigh(left_balls, right_balls)
                steps += 1
                self.animate_weighing(left_balls, right_balls, result3)

                if result3 == "balanced":
                    self.result_label.config(text="结果：球 11 较重", fg="red")
                    self.mark_ball(11, "red", "较重")
                elif result3 == "left_heavy":
                    self.result_label.config(text="结果：球 9 较重", fg="red")
                    self.mark_ball(9, "red", "较重")
                else:
                    self.result_label.config(text="结果：球 10 较重", fg="red")
                    self.mark_ball(10, "red", "较重")

            else:  # right_heavy
                self.info_label.config(text="第二次结果：右盘重 → 坏球在 9-11 中且较轻")
                self.root.update()
                time.sleep(1)

                # 第三次：9 vs 10
                self.info_label.config(text="第三次称重：比较球 9 和 球 10")
                left_balls = [9]
                right_balls = [10]
                result3 = self.weigh(left_balls, right_balls)
                steps += 1
                self.animate_weighing(left_balls, right_balls, result3)

                if result3 == "balanced":
                    self.result_label.config(text="结果：球 11 较轻", fg="blue")
                    self.mark_ball(11, "lightblue", "较轻")
                elif result3 == "left_heavy":
                    self.result_label.config(text="结果：球 10 较轻", fg="blue")
                    self.mark_ball(10, "lightblue", "较轻")
                else:
                    self.result_label.config(text="结果：球 9 较轻", fg="blue")
                    self.mark_ball(9, "lightblue", "较轻")

        elif result1 == "left_heavy":
            # 坏球在1-8中：球1-4可能重，球5-8可能轻
            self.info_label.config(text="第一次结果：左盘重 → 坏球在球 1-8 中")
            self.root.update()
            time.sleep(1)

            # 第二次：1,2,5 vs 3,6,9（9已知正常）
            self.info_label.config(text="第二次称重：比较球 1,2,5 和 球 3,6,9（已知正常）")
            left_balls = [1, 2, 5]
            right_balls = [3, 6, 9]
            result2 = self.weigh(left_balls, right_balls)
            steps += 1
            self.animate_weighing(left_balls, right_balls, result2)

            if result2 == "balanced":
                # 坏球在4,7,8中：球4可能重，球7,8可能轻
                self.info_label.config(text="第二次结果：平衡 → 坏球在球 4（重）或球 7,8（轻）中")
                self.root.update()
                time.sleep(1)

                self.info_label.config(text="第三次称重：比较球 7 和 球 8")
                left_balls = [7]
                right_balls = [8]
                result3 = self.weigh(left_balls, right_balls)
                steps += 1
                self.animate_weighing(left_balls, right_balls, result3)

                if result3 == "balanced":
                    self.result_label.config(text="结果：球 4 较重", fg="red")
                    self.mark_ball(4, "red", "较重")
                elif result3 == "left_heavy":
                    self.result_label.config(text="结果：球 8 较轻", fg="blue")
                    self.mark_ball(8, "lightblue", "较轻")
                else:
                    self.result_label.config(text="结果：球 7 较轻", fg="blue")
                    self.mark_ball(7, "lightblue", "较轻")

            elif result2 == "left_heavy":
                # 坏球在1,2中（重）或6中（轻）
                self.info_label.config(text="第二次结果：左盘重 → 坏球在球 1,2（重）或球 6（轻）中")
                self.root.update()
                time.sleep(1)

                self.info_label.config(text="第三次称重：比较球 1 和 球 2")
                left_balls = [1]
                right_balls = [2]
                result3 = self.weigh(left_balls, right_balls)
                steps += 1
                self.animate_weighing(left_balls, right_balls, result3)

                if result3 == "balanced":
                    self.result_label.config(text="结果：球 6 较轻", fg="blue")
                    self.mark_ball(6, "lightblue", "较轻")
                elif result3 == "left_heavy":
                    self.result_label.config(text="结果：球 1 较重", fg="red")
                    self.mark_ball(1, "red", "较重")
                else:
                    self.result_label.config(text="结果：球 2 较重", fg="red")
                    self.mark_ball(2, "red", "较重")

            else:  # right_heavy
                # 坏球在5中（轻）或3中（重）
                self.info_label.config(text="第二次结果：右盘重 → 坏球在球 5（轻）或球 3（重）中")
                self.root.update()
                time.sleep(1)

                self.info_label.config(text="第三次称重：比较球 5 和 球 9（已知正常）")
                left_balls = [5]
                right_balls = [9]
                result3 = self.weigh(left_balls, right_balls)
                steps += 1
                self.animate_weighing(left_balls, right_balls, result3)

                if result3 == "balanced":
                    self.result_label.config(text="结果：球 3 较重", fg="red")
                    self.mark_ball(3, "red", "较重")
                elif result3 == "left_heavy":
                    self.result_label.config(text="结果：球 5 较重（异常）", fg="red")
                    self.mark_ball(5, "red", "较重")
                else:
                    self.result_label.config(text="结果：球 5 较轻", fg="blue")
                    self.mark_ball(5, "lightblue", "较轻")

        else:  # result1 == "right_heavy"
            # 坏球在1-8中：球1-4可能轻，球5-8可能重
            self.info_label.config(text="第一次结果：右盘重 → 坏球在球 1-8 中")
            self.root.update()
            time.sleep(1)

            # 第二次：1,2,5 vs 3,6,9（9已知正常）
            self.info_label.config(text="第二次称重：比较球 1,2,5 和 球 3,6,9（已知正常）")
            left_balls = [1, 2, 5]
            right_balls = [3, 6, 9]
            result2 = self.weigh(left_balls, right_balls)
            steps += 1
            self.animate_weighing(left_balls, right_balls, result2)

            if result2 == "balanced":
                # 坏球在4,7,8中：球4可能轻，球7,8可能重
                self.info_label.config(text="第二次结果：平衡 → 坏球在球 4（轻）或球 7,8（重）中")
                self.root.update()
                time.sleep(1)

                self.info_label.config(text="第三次称重：比较球 7 和 球 8")
                left_balls = [7]
                right_balls = [8]
                result3 = self.weigh(left_balls, right_balls)
                steps += 1
                self.animate_weighing(left_balls, right_balls, result3)

                if result3 == "balanced":
                    self.result_label.config(text="结果：球 4 较轻", fg="blue")
                    self.mark_ball(4, "lightblue", "较轻")
                elif result3 == "left_heavy":
                    self.result_label.config(text="结果：球 7 较重", fg="red")
                    self.mark_ball(7, "red", "较重")
                else:
                    self.result_label.config(text="结果：球 8 较重", fg="red")
                    self.mark_ball(8, "red", "较重")

            elif result2 == "left_heavy":
                # 坏球在5中（重）或3中（轻）
                self.info_label.config(text="第二次结果：左盘重 → 坏球在球 5（重）或球 3（轻）中")
                self.root.update()
                time.sleep(1)

                self.info_label.config(text="第三次称重：比较球 5 和 球 9（已知正常）")
                left_balls = [5]
                right_balls = [9]
                result3 = self.weigh(left_balls, right_balls)
                steps += 1
                self.animate_weighing(left_balls, right_balls, result3)

                if result3 == "balanced":
                    self.result_label.config(text="结果：球 3 较轻", fg="blue")
                    self.mark_ball(3, "lightblue", "较轻")
                elif result3 == "left_heavy":
                    self.result_label.config(text="结果：球 5 较重", fg="red")
                    self.mark_ball(5, "red", "较重")
                else:
                    self.result_label.config(text="结果：球 5 较轻", fg="blue")
                    self.mark_ball(5, "lightblue", "较轻")

            else:  # right_heavy
                # 坏球在1,2中（轻）或6中（重）
                self.info_label.config(text="第二次结果：右盘重 → 坏球在球 1,2（轻）或球 6（重）中")
                self.root.update()
                time.sleep(1)

                self.info_label.config(text="第三次称重：比较球 1 和 球 2")
                left_balls = [1]
                right_balls = [2]
                result3 = self.weigh(left_balls, right_balls)
                steps += 1
                self.animate_weighing(left_balls, right_balls, result3)

                if result3 == "balanced":
                    self.result_label.config(text="结果：球 6 较重", fg="red")
                    self.mark_ball(6, "red", "较重")
                elif result3 == "left_heavy":
                    self.result_label.config(text="结果：球 2 较轻", fg="blue")
                    self.mark_ball(2, "lightblue", "较轻")
                else:
                    self.result_label.config(text="结果：球 1 较轻", fg="blue")
                    self.mark_ball(1, "lightblue", "较轻")

        self.info_label.config(text=f"演示完成！共进行 {steps} 次称重找到坏球")
        self.animation_running = False
        self.start_btn.config(state=tk.NORMAL)

    def reset(self):
        """重置演示"""
        self.canvas.delete("all")
        self.balls = [0] * 12
        self.bad_ball = random.randint(0, 11)
        self.bad_ball_type = random.choice([1, -1])
        self.balls[self.bad_ball] = self.bad_ball_type

        self.draw_balls()
        self.draw_scale()
        self.info_label.config(text="")
        self.result_label.config(text="")
        self.start_btn.config(state=tk.NORMAL)
        self.animation_running = False

    def run(self):
        self.root.mainloop()

def main():
    app = TwelveBallPuzzle()
    app.run()

if __name__ == "__main__":
    main()
