"""
将目录下所有 .py 文件合并为一个文本文件，便于代码汇总查阅。
教学重点：Python pathlib 文件路径操作、目录递归遍历、文本文件读写。
"""

from pathlib import Path

# --- 1. 工具函数 ---
def normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")

# --- 2. 主逻辑：定位目录、收集 .py 文件、合并输出 ---
def main():
    # deep.py 所在目录：E:\...\深度学习\某个子目录
    here = Path(__file__).resolve().parent

    # 你的目录结构：项目根\深度学习\xxxx\deep.py
    # 所以“深度学习”目录就是 here 的父目录
    dl_dir = here.parent  # E:\代码\minicondapythonProject1\深度学习

    # 再保险一下：确认目录名确实是“深度学习”
    if dl_dir.name != "深度学习":
        raise FileNotFoundError(
            f"定位失败：推断的深度学习目录是 {dl_dir}，但目录名不是“深度学习”。\n"
            f"请检查 deep.py 是否放在 深度学习 的子目录下。"
        )

    # 只收集“当前这个子目录”（截图里那个乱码目录）下的 .py
    # 如果你想收集整个“深度学习”目录，把这里改成：dl_dir.rglob("*.py")
    target_dir = here

    py_files = sorted(
        [f for f in target_dir.rglob("*.py")
         if f.is_file()
         and "__pycache__" not in f.parts
         and ".idea" not in f.parts
         and ".git" not in f.parts],
        key=lambda x: x.relative_to(target_dir).as_posix()
    )

    if not py_files:
        raise RuntimeError(f"目录 {target_dir} 下没有找到任何 .py 文件。")

    out_path = target_dir / "深度学习代码.txt"

    with out_path.open("w", encoding="utf-8") as out:
        for f in py_files:
            rel = f.relative_to(target_dir).as_posix()
            out.write(f"========== {rel} ==========\n")

            text = f.read_text(encoding="utf-8", errors="replace")
            text = normalize_newlines(text).rstrip("\n")
            out.write(text + "\n")

    print(f"✅ 已生成：{out_path}")

if __name__ == "__main__":
    main()
