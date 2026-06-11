#!/usr/bin/env python3
"""
TF-IDF 原理动画演示
TF (词频) = 词在文档中出现次数 / 文档总词数
IDF (逆文档频率) = log(总文档数 / 包含该词的文档数)
TF-IDF = TF × IDF
"""

import tkinter as tk
import math
import time
import threading


class TFIDFDemo:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("TF-IDF 原理动画演示")
        self.root.geometry("1100x750")
        self.root.configure(bg="#f0f0f0")

        # 示例语料库
        self.corpus = [
            ["猫", "狗", "猫", "鱼", "猫"],
            ["狗", "狗", "骨头", "猫", "狗"],
            ["鱼", "猫", "鱼", "鱼", "水"],
        ]
        self.doc_names = ["文档A", "文档B", "文档C"]
        self.target_word = "猫"

        self.step = 0
        self.animating = False
        self.setup_ui()
        self.draw_initial()

    def setup_ui(self):
        # 标题
        tk.Label(self.root, text="TF-IDF 原理动画演示",
                 font=("Arial", 20, "bold"), bg="#f0f0f0").pack(pady=8)

        # 主画布
        self.canvas = tk.Canvas(self.root, width=1060, height=550, bg="white",
                                highlightthickness=1, highlightbackground="#ccc")
        self.canvas.pack(pady=5)

        # 按钮区
        btn_frame = tk.Frame(self.root, bg="#f0f0f0")
        btn_frame.pack(pady=8)

        self.btn_start = tk.Button(btn_frame, text="开始演示", font=("Arial", 13, "bold"),
                                   bg="#4CAF50", fg="white", width=10, command=self.start_demo)
        self.btn_start.pack(side=tk.LEFT, padx=6)

        self.btn_next = tk.Button(btn_frame, text="下一步", font=("Arial", 13),
                                  width=8, command=self.next_step, state=tk.DISABLED)
        self.btn_next.pack(side=tk.LEFT, padx=6)

        self.btn_reset = tk.Button(btn_frame, text="重置", font=("Arial", 13),
                                   width=6, command=self.reset)
        self.btn_reset.pack(side=tk.LEFT, padx=6)

        # 状态栏
        self.status = tk.Label(self.root, text="点击「开始演示」观看 TF-IDF 计算过程",
                               font=("Arial", 12), bg="#f0f0f0", fg="#555")
        self.status.pack(pady=4)

    # ── 绘图辅助 ──────────────────────────────────────────────

    def _rounded_rect(self, x1, y1, x2, y2, r, **kw):
        pts = [x1+r, y1, x2-r, y1, x2, y1, x2, y1+r, x2, y2-r, x2, y2,
               x2-r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y1+r, x1, y1]
        return self.canvas.create_polygon(pts, smooth=True, **kw)

    def draw_initial(self):
        """绘制初始语料库"""
        self.canvas.delete("all")

        # 左侧标题
        self.canvas.create_text(30, 20, text="语料库（3 篇文档）",
                                font=("Arial", 14, "bold"), anchor="w", fill="#333")

        # 绘制三篇文档
        colors = ["#E3F2FD", "#FFF3E0", "#E8F5E9"]
        borders = ["#1976D2", "#F57C00", "#388E3C"]
        for di, (doc, name, col, bor) in enumerate(
                zip(self.corpus, self.doc_names, colors, borders)):
            y0 = 50 + di * 140
            self._rounded_rect(20, y0, 480, y0 + 120, 10,
                               fill=col, outline=bor, width=2)
            self.canvas.create_text(35, y0 + 15, text=name,
                                    font=("Arial", 12, "bold"), anchor="w", fill=bor)
            # 词袋
            for wi, word in enumerate(doc):
                wx = 40 + wi * 85
                wy = y0 + 70
                self._rounded_rect(wx - 28, wy - 18, wx + 28, wy + 18, 8,
                                   fill="white", outline="#999", width=1,
                                   tags=f"word_{di}_{wi}")
                self.canvas.create_text(wx, wy, text=word,
                                        font=("Arial", 14), tags=f"wtxt_{di}_{wi}")

        # 右侧公式说明区
        self.canvas.create_text(540, 20, text="计算公式",
                                font=("Arial", 14, "bold"), anchor="w", fill="#333")
        formulas = [
            "TF(t, d) = 词t在文档d中的出现次数 / 文档d的总词数",
            "IDF(t)   = log( 文档总数 / 包含词t的文档数 )",
            "TF-IDF   = TF × IDF",
        ]
        for i, f in enumerate(formulas):
            self.canvas.create_text(540, 55 + i * 30, text=f,
                                    font=("Consolas", 11), anchor="w", fill="#444")

        # 高亮目标词
        self.canvas.create_text(540, 160,
                                text=f'目标词："{self.target_word}"',
                                font=("Arial", 14, "bold"), anchor="w", fill="#D32F2F")

        # 右侧结果区占位
        self.canvas.create_text(540, 200, text="计算结果将在此处显示",
                                font=("Arial", 11), anchor="w", fill="#aaa",
                                tags="result_placeholder")

    # ── 动画核心 ──────────────────────────────────────────────

    def _highlight_word(self, doc_idx, word_idx, color):
        tag = f"word_{doc_idx}_{word_idx}"
        self.canvas.itemconfig(tag, fill=color, outline=color, width=2)

    def _highlight_target_words(self, color):
        """高亮所有目标词"""
        for di, doc in enumerate(self.corpus):
            for wi, word in enumerate(doc):
                if word == self.target_word:
                    self._highlight_word(di, wi, color)
        self.root.update()

    def _flash_word(self, doc_idx, word_idx, color, times=2):
        """闪烁某个词"""
        tag = f"word_{doc_idx}_{word_idx}"
        for _ in range(times):
            self.canvas.itemconfig(tag, fill=color)
            self.root.update()
            time.sleep(0.15)
            self.canvas.itemconfig(tag, fill="white")
            self.root.update()
            time.sleep(0.1)

    def _draw_tf_table(self):
        """绘制TF计算表"""
        self.canvas.delete("result")
        self.canvas.delete("result_placeholder")

        x0, y0 = 520, 195

        self._rounded_rect(x0 - 10, y0 - 5, x0 + 520, y0 + 28, 6,
                           fill="#E8EAF6", outline="#5C6BC0", width=1, tags="result")
        self.canvas.create_text(x0, y0 + 12, text="Step 1: 计算 TF（词频）",
                                font=("Arial", 13, "bold"), anchor="w",
                                fill="#283593", tags="result")

        y = y0 + 50
        for di, (doc, name) in enumerate(zip(self.corpus, self.doc_names)):
            count = doc.count(self.target_word)
            total = len(doc)
            tf = count / total

            # 文档标签
            self.canvas.create_text(x0, y, text=f"{name}:", font=("Arial", 12, "bold"),
                                    anchor="w", fill="#333", tags="result")
            # 词计数
            self.canvas.create_text(x0 + 60, y,
                                    text=f'"{self.target_word}"出现 {count} 次，共 {total} 词',
                                    font=("Arial", 11), anchor="w", fill="#555", tags="result")
            # TF 值
            self.canvas.create_text(x0 + 310, y,
                                    text=f"TF = {count}/{total} = {tf:.3f}",
                                    font=("Consolas", 12, "bold"), anchor="w",
                                    fill="#1565C0", tags="result")
            y += 30

        return y + 10

    def _draw_idf_table(self, y_start):
        """绘制IDF计算表"""
        x0 = 520
        y = y_start + 10

        self._rounded_rect(x0 - 10, y - 5, x0 + 520, y + 28, 6,
                           fill="#FFF3E0", outline="#FF9800", width=1, tags="result")
        self.canvas.create_text(x0, y + 12, text="Step 2: 计算 IDF（逆文档频率）",
                                font=("Arial", 13, "bold"), anchor="w",
                                fill="#E65100", tags="result")

        y += 50
        total_docs = len(self.corpus)
        contain_count = sum(1 for doc in self.corpus if self.target_word in doc)
        idf = math.log(total_docs / contain_count)

        self.canvas.create_text(x0, y, text=f"文档总数 N = {total_docs}",
                                font=("Arial", 12), anchor="w", fill="#555", tags="result")
        y += 28
        self.canvas.create_text(x0, y,
                                text=f'包含"{self.target_word}"的文档数 = {contain_count}',
                                font=("Arial", 12), anchor="w", fill="#555", tags="result")
        y += 28
        self.canvas.create_text(x0, y,
                                text=f"IDF = log({total_docs}/{contain_count}) = log({total_docs}/{contain_count}) = {idf:.4f}",
                                font=("Consolas", 12, "bold"), anchor="w",
                                fill="#E65100", tags="result")
        y += 28
        self.canvas.create_text(x0 + 10, y,
                                text="(词越罕见 → IDF越大 → 该词区分度越高)",
                                font=("Arial", 10), anchor="w", fill="#999", tags="result")

        return y + 20

    def _draw_tfidf_table(self, y_start):
        """绘制TF-IDF结果表"""
        x0 = 520
        y = y_start + 10

        self._rounded_rect(x0 - 10, y - 5, x0 + 520, y + 28, 6,
                           fill="#E8F5E9", outline="#4CAF50", width=1, tags="result")
        self.canvas.create_text(x0, y + 12, text="Step 3: 计算 TF-IDF",
                                font=("Arial", 13, "bold"), anchor="w",
                                fill="#1B5E20", tags="result")

        y += 50
        total_docs = len(self.corpus)
        contain_count = sum(1 for doc in self.corpus if self.target_word in doc)
        idf = math.log(total_docs / contain_count)

        max_tfidf = 0
        for di, (doc, name) in enumerate(zip(self.corpus, self.doc_names)):
            count = doc.count(self.target_word)
            tf = count / len(doc)
            tfidf = tf * idf
            max_tfidf = max(max_tfidf, tfidf)

            self.canvas.create_text(x0, y, text=f"{name}:", font=("Arial", 12, "bold"),
                                    anchor="w", fill="#333", tags="result")
            self.canvas.create_text(x0 + 60, y,
                                    text=f"TF-IDF = {tf:.3f} × {idf:.4f} = {tfidf:.4f}",
                                    font=("Consolas", 12, "bold"), anchor="w",
                                    fill="#2E7D32", tags="result")

            # 绘制柱状图
            bar_x = x0 + 340
            bar_w = max(20, int(tfidf / max_tfidf * 130)) if max_tfidf > 0 else 20
            bar_colors = ["#42A5F5", "#FFA726", "#66BB6A"]
            self._rounded_rect(bar_x, y - 10, bar_x + bar_w, y + 10, 4,
                               fill=bar_colors[di], outline="", tags="result")
            y += 35

        # 总结
        y += 15
        self._rounded_rect(x0 - 10, y - 5, x0 + 520, y + 55, 8,
                           fill="#FCE4EC", outline="#E91E63", width=2, tags="result")
        self.canvas.create_text(x0 + 5, y + 8,
                                text="结论：TF-IDF 越高 → 该词对这篇文档越重要/有代表性",
                                font=("Arial", 12, "bold"), anchor="w",
                                fill="#AD1457", tags="result")
        self.canvas.create_text(x0 + 5, y + 35,
                                text="常见词（如"的""是"）在所有文档都出现 → IDF低 → TF-IDF低",
                                font=("Arial", 11), anchor="w", fill="#888", tags="result")

    # ── 步骤控制 ──────────────────────────────────────────────

    def _set_status(self, text):
        self.status.config(text=text)
        self.root.update()

    def next_step(self):
        if self.animating:
            return
        self.animating = True
        threading.Thread(target=self._run_step, daemon=True).start()

    def _run_step(self):
        self.step += 1

        if self.step == 1:
            # Step 1: 高亮目标词 + 计算TF
            self._set_status("Step 1: 高亮目标词「猫」，统计每个文档中的词频 (TF)")
            self._highlight_target_words("#FFEB3B")
            time.sleep(0.8)

            # 逐个闪烁每个文档中的目标词
            for di, doc in enumerate(self.corpus):
                for wi, word in enumerate(doc):
                    if word == self.target_word:
                        self._flash_word(di, wi, "#FF9800")
            time.sleep(0.3)

            self._draw_tf_table()
            self._highlight_target_words("#FFEB3B")

        elif self.step == 2:
            # Step 2: 计算IDF
            self._set_status("Step 2: 计算 IDF —— 统计有多少篇文档包含「猫」")

            # 逐篇文档高亮
            colors = ["#1976D2", "#F57C00", "#388E3C"]
            for di, doc in enumerate(self.corpus):
                y0 = 50 + di * 140
                if self.target_word in doc:
                    self._set_status(f'Step 2: {self.doc_names[di]} 包含「猫」✓')
                    # 闪烁文档边框
                    for _ in range(2):
                        self.canvas.create_rectangle(18, y0 - 2, 482, y0 + 122,
                                                     outline="red", width=3, tags="flash")
                        self.root.update()
                        time.sleep(0.2)
                        self.canvas.delete("flash")
                        self.root.update()
                        time.sleep(0.15)
                else:
                    self._set_status(f'Step 2: {self.doc_names[di]} 不包含「猫」✗')
                time.sleep(0.3)

            y = self._draw_tf_table()
            self._draw_idf_table(y)
            self._highlight_target_words("#FFEB3B")

        elif self.step == 3:
            # Step 3: 计算TF-IDF
            self._set_status("Step 3: TF × IDF = TF-IDF，得到每个文档中「猫」的重要性得分")

            y = self._draw_tf_table()
            y = self._draw_idf_table(y)
            self._draw_tfidf_table(y)

            # 最终高亮：按TF-IDF大小着色
            total_docs = len(self.corpus)
            contain_count = sum(1 for d in self.corpus if self.target_word in d)
            idf = math.log(total_docs / contain_count)
            heat_colors = ["#BBDEFB", "#FFF9C4", "#C8E6C9"]
            for di, doc in enumerate(self.corpus):
                count = doc.count(self.target_word)
                tf = count / len(doc)
                tfidf = tf * idf
                intensity = int(min(255, tfidf / 0.2 * 200))
                for wi, word in enumerate(doc):
                    if word == self.target_word:
                        r = 255
                        g = max(0, 255 - intensity)
                        b = max(0, 255 - intensity)
                        color = f"#{r:02x}{g:02x}{b:02x}"
                        tag = f"word_{di}_{wi}"
                        self.canvas.itemconfig(tag, fill=color)
            self.root.update()
            self._set_status("演示完成！红色越深 → TF-IDF越高 → 该词对该文档越重要")
            self.btn_next.config(state=tk.DISABLED)

        self.animating = False

    def start_demo(self):
        """自动播放全部步骤"""
        if self.animating:
            return
        self.btn_start.config(state=tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL)

        # 如果还没开始，先执行step 1
        if self.step == 0:
            self.next_step()

        # 然后用定时器自动推进
        self.root.after(3000, self._auto_next)

    def _auto_next(self):
        if self.step < 3 and not self.animating:
            self.next_step()
            self.root.after(3500, self._auto_next)
        elif self.step >= 3:
            self.btn_start.config(state=tk.NORMAL)

    def next_step(self):
        if self.animating:
            return
        if self.step >= 3:
            return
        self.animating = True
        threading.Thread(target=self._run_step, daemon=True).start()

    def reset(self):
        self.canvas.delete("all")
        self.step = 0
        self.animating = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_next.config(state=tk.DISABLED)
        self.status.config(text="点击「开始演示」观看 TF-IDF 计算过程")
        self.draw_initial()

    def run(self):
        self.root.mainloop()


def main():
    app = TFIDFDemo()
    app.run()


if __name__ == "__main__":
    main()
