# -*- coding: utf-8 -*-
"""
八皇后问题解决方案 (优化美化版)
==============================
经典国际象棋问题：在8×8棋盘上放置8个皇后
使得任意两个皇后都不能互相攻击（不在同行、同列、同对角线）

使用回溯算法求解所有92个唯一解，并美观展示
"""

import time


class Colors:
    """ANSI颜色代码类"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # 前景色
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # 高强度前景色
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    
    # 背景色
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'


def print_banner():
    """打印程序横幅"""
    banner = f"""
{Colors.BOLD}{Colors.BRIGHT_CYAN}{'='*70}
   ♛  八皇后问题求解器 (优化美化版)  ♛
{'='*70}{Colors.RESET}

{Colors.YELLOW}问题描述：
  在8×8的国际象棋棋盘上放置8个皇后
  使得任意两个皇后都不能互相攻击

攻击规则：
  皇后可以攻击同一行、同一列、同一对角线上的所有格子
  因此8个皇后必须满足：不在同一行、不在同一列、不在同一对角线{Colors.RESET}

{Colors.CYAN}算法：
  使用回溯法(Backtracking)系统地搜索所有可能解
  从第1列开始，逐列尝试放置皇后
  如果后续无法放置，就回溯到上一步{Colors.RESET}

"""
    print(banner)


def is_safe(board, row, col):
    """
    检查皇后是否可以放在board[row][col]
    不会被其他皇后攻击
    """
    # 检查左侧同一行
    for j in range(col):
        if board[row][j] == 1:
            return False
    
    # 检查左上方对角线
    i, j = row, col
    while i >= 0 and j >= 0:
        if board[i][j] == 1:
            return False
        i -= 1
        j -= 1
    
    # 检查左下方对角线
    i, j = row, col
    while i < 8 and j >= 0:
        if board[i][j] == 1:
            return False
        i += 1
        j -= 1
    
    return True


def solve_queens(board, col, solutions):
    """
    使用回溯法求解八皇后问题
    """
    # 基础情况：所有皇后都已放置
    if col >= 8:
        solution = [row[:] for row in board]
        solutions.append(solution)
        return True
    
    # 尝试在当前列的每一行放置皇后
    for row in range(8):
        if is_safe(board, row, col):
            # 放置皇后
            board[row][col] = 1
            
            # 递归处理下一列
            solve_queens(board, col + 1, solutions)
            
            # 回溯
            board[row][col] = 0
    
    return True


def display_board_colored(board, solution_num, total_solutions):
    """
    彩色美化显示棋盘
    """
    col_width = 5
    
    # 打印标题
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}{'━'*60}")
    print(f"  解 #{solution_num}/{total_solutions}")
    print(f"{'━'*60}{Colors.RESET}\n")
    
    # 列标签 (A-H)
    print(f"    ", end="")
    for col in range(8):
        label = chr(65 + col)
        padding = (col_width - 1) // 2
        print(f"{' ' * padding}{Colors.BOLD}{label}{Colors.RESET}{' ' * (col_width - padding - 1)}", end="")
    print("\n")
    
    # 上边框
    print(f"    {Colors.DIM}┌{('─' * (col_width - 2) + '┬') * 7}{'─' * (col_width - 2)}┐{Colors.RESET}")
    
    # 棋盘行
    for row in range(8):
        row_label = 8 - row
        
        # 左边框和行号
        print(f"  {Colors.BOLD}{row_label}{Colors.RESET} {Colors.DIM}│{Colors.RESET}", end="")
        
        # 每一列
        for col in range(8):
            if board[row][col] == 1:
                # 皇后位置 - 使用醒目的背景色
                bg_color = Colors.BG_MAGENTA if (row + col) % 2 == 0 else Colors.BG_BLUE
                queen = f"{Colors.BOLD}{Colors.WHITE}{bg_color} ♛  {Colors.RESET}"
                print(queen, end="")
            else:
                # 空位 - 交替背景色模拟棋盘格
                bg_color = Colors.BG_WHITE if (row + col) % 2 == 0 else Colors.BG_BLACK
                print(f"{bg_color}{' ' * (col_width - 2)}{Colors.RESET}", end="")
            
            # 列分隔线
            if col < 7:
                print(f"{Colors.DIM}│{Colors.RESET}", end="")
        
        # 右边框和行号
        print(f"{Colors.DIM}│{Colors.RESET}  {Colors.BOLD}{row_label}{Colors.RESET}")
        
        # 行分隔线
        if row < 7:
            print(f"    {Colors.DIM}├{('─' * (col_width - 2) + '┼') * 7}{'─' * (col_width - 2)}┤{Colors.RESET}")
    
    # 下边框
    print(f"    {Colors.DIM}└{('─' * (col_width - 2) + '┴') * 7}{'─' * (col_width - 2)}┘{Colors.RESET}")
    
    # 列标签 (A-H)
    print(f"    ", end="")
    for col in range(8):
        label = chr(65 + col)
        padding = (col_width - 1) // 2
        print(f"{' ' * padding}{Colors.BOLD}{label}{Colors.RESET}{' ' * (col_width - padding - 1)}", end="")
    print("\n")


def print_statistics(solutions, elapsed_time):
    """打印统计信息"""
    stats = f"""
{Colors.BOLD}{Colors.BRIGHT_GREEN}{'='*70}
   📊 统计结果
{'='*70}{Colors.RESET}

{Colors.CYAN}总解数：        {Colors.BOLD}{len(solutions)}{Colors.RESET} 个唯一解
{Colors.CYAN}求解时间：      {Colors.BOLD}{elapsed_time:.4f}{Colors.RESET} 秒
{Colors.CYAN}平均速度：      {Colors.BOLD}{len(solutions)/elapsed_time:.0f}{Colors.RESET} 解/秒

{Colors.YELLOW}算法特点：
  • 使用回溯法系统搜索
  • 时间复杂度：O(N!)
  • 空间复杂度：O(N)
  • 保证找到所有解{Colors.RESET}

{Colors.MAGENTA}有趣的数学事实：
  • 八皇后问题共有92个唯一解
  • 如果考虑旋转和镜像，只有12个本质不同的解
  • N皇后问题在N=1,2,3时无解{Colors.RESET}

{Colors.BOLD}{Colors.BRIGHT_CYAN}{'='*70}
   ✓ 所有解已成功展示
{'='*70}{Colors.RESET}
"""
    print(stats)


def main():
    """主函数"""
    # 打印横幅
    print_banner()
    
    # 初始化空棋盘
    board = [[0 for _ in range(8)] for _ in range(8)]
    solutions = []
    
    # 进度显示
    print(f"\n{Colors.YELLOW}🔄 开始求解八皇后问题...{Colors.RESET}")
    print(f"{Colors.DIM}正在使用回溯算法搜索所有可能解...{Colors.RESET}\n")
    
    # 记录开始时间
    start_time = time.time()
    
    # 求解所有解
    solve_queens(board, 0, solutions)
    
    # 记录结束时间
    elapsed_time = time.time() - start_time
    
    print(f"{Colors.GREEN}✓ 找到所有 {len(solutions)} 个解！{Colors.RESET}")
    print()
    
    # 打印统计信息
    print_statistics(solutions, elapsed_time)
    
    # 显示所有92个解
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}{'='*70}")
    print("   🎯 所有92个解的展示")
    print(f"{'='*70}{Colors.RESET}\n")
    
    # 分批显示解
    batch_size = 5
    for batch_start in range(0, len(solutions), batch_size):
        batch_end = min(batch_start + batch_size, len(solutions))
        
        # 打印批次标题
        print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}{'─'*60}")
        print(f"  解 #{batch_start + 1} - #{batch_end}")
        print(f"{'─'*60}{Colors.RESET}")
        
        # 显示批次中的每个解
        for i in range(batch_start, batch_end):
            display_board_colored(solutions[i], i + 1, len(solutions))
            
            # 在解之间添加分隔线
            if i < batch_end - 1:
                print(f"\n{Colors.DIM}{'·'*60}{Colors.RESET}\n")
    
    # 最终总结
    print(f"\n\n{Colors.BOLD}{Colors.BRIGHT_GREEN}{'='*70}")
    print("   ✅ 展示完成！")
    print(f"{'='*70}{Colors.RESET}")
    print(f"\n{Colors.CYAN}共展示了 {Colors.BOLD}{len(solutions)}{Colors.RESET}{Colors.CYAN} 个唯一解")
    print(f"每个解都满足八皇后问题的所有约束条件{Colors.RESET}\n")
    
    # 解释如何阅读
    print(f"{Colors.YELLOW}📖 阅读说明：")
    print(f"  ♛ = 皇后位置")
    print(f"  行号 (1-8) 从上到下")
    print(f"  列号 (A-H) 从左到右")
    print(f"  背景色区分格子颜色 (浅色/深色){Colors.RESET}\n")


if __name__ == "__main__":
    main()
