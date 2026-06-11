"""
数据采集与处理实验 - 代码汇总工具

自动扫描 experiment1~10 目录下的所有代码文件，汇总为一个完整的文本文件。
教学重点：文件系统遍历、正则匹配与文本处理的综合应用。
"""
from pathlib import Path
import re

# --- 1. 全局配置 ---
# 课程实验所在目录名（按实际项目结构修改）
COURSE_DIR_NAME = "数据采集与处理实验"

# 需要收集的代码文件后缀
CODE_EXTS = {".py", ".html", ".htm", ".js", ".css", ".json", ".txt", ".md"}


def normalize_newlines(s: str) -> str:
    """统一换行符，避免 Windows/Linux 格式混乱"""
    return s.replace("\r\n", "\n").replace("\r", "\n")


def is_experiment_dir(p: Path) -> bool:
    """判断目录名是否为实验目录（支持 experiment1 ~ experiment10）"""
    m = re.fullmatch(r"experiment0?([1-9]|10)", p.name, flags=re.IGNORECASE)
    return bool(m)


def exp_number(name: str) -> int:
    """从实验目录名中提取实验编号"""
    m = re.fullmatch(r"experiment0?([1-9]|10)", name, flags=re.IGNORECASE)
    return int(m.group(1))


def collect_files(exp_dir: Path):
    """递归收集实验目录下所有指定后缀的代码文件，排除缓存和IDE目录"""
    files = []
    for f in exp_dir.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in CODE_EXTS:
            continue
        # 排除常见垃圾目录
        if any(part in {"__pycache__", ".idea", ".git", ".vscode"} for part in f.parts):
            continue
        files.append(f)
    # 按相对路径排序，稳定输出
    return sorted(files, key=lambda x: x.relative_to(exp_dir).as_posix())

# --- 2. 主流程 ---
def main():
    """扫描 experiment1~10 目录，将所有代码文件汇总输出为一个文本文件"""
    project_root = Path.cwd()  # 直接在PyCharm运行时一般就是项目根
    course_dir = project_root / COURSE_DIR_NAME
    if not course_dir.exists():
        raise FileNotFoundError(f"找不到目录：{course_dir}\n"
                                f"请确认你是在项目根目录运行，或修改 COURSE_DIR_NAME。")

    # 找到 experiment1~10 文件夹
    exp_dirs = [p for p in course_dir.iterdir() if p.is_dir() and is_experiment_dir(p)]
    exp_dirs = sorted(exp_dirs, key=lambda p: exp_number(p.name))

    # 只保留1~10
    exp_dirs = [p for p in exp_dirs if 1 <= exp_number(p.name) <= 10]

    if not exp_dirs:
        raise RuntimeError("没有找到 experiment1~10 相关目录（例如 experiment03 / experiment10）。")

    out_path = course_dir / "实验1-10代码汇总.txt"

    with out_path.open("w", encoding="utf-8") as out:
        for idx, exp_dir in enumerate(exp_dirs):
            n = exp_number(exp_dir.name)

            # 实验标题
            out.write(f"========== 实验{n}（{exp_dir.name}） ==========\n")

            files = collect_files(exp_dir)
            if not files:
                out.write("[该实验目录下未发现指定后缀的代码文件]\n")
            else:
                for f in files:
                    rel = f.relative_to(exp_dir).as_posix()
                    out.write(f"\n--- 文件：{rel} ---\n")  # 这是“正常换行”，不是空行分隔
                    try:
                        text = f.read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        # 有些文件可能不是utf-8，兜底用二进制再解
                        text = f.read_bytes().decode("utf-8", errors="replace")
                    text = normalize_newlines(text).rstrip("\n")
                    out.write(text + "\n")

            # 实验与实验之间空一行（只有这里加“空行”）
            if idx != len(exp_dirs) - 1:
                out.write("\n")

    print(f"✅ 已生成：{out_path}")

if __name__ == "__main__":
    main()
