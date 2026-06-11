"""
教学示例：词云可视化

- 功能：演示如何把一段文本统计成词频，再生成最基础的中文词云图。
- 主要数据结构：使用字符串保存文本，使用字典保存词频映射。
- 这样设置的原因：把流程压缩成一个最小可运行示例，便于零基础同学先理解“文本 -> 词频 -> 图片”的链路。
"""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from wordcloud import WordCloud

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "wordcloud_demo.png"
FONT_CANDIDATES = [
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    Path("/System/Library/Fonts/Supplemental/Songti.ttc"),
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/System/Library/Fonts/STHeiti Light.ttc"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
]


def build_demo_text() -> str:
    """返回一段适合课堂演示的样本文本。"""
    return "计算机视觉 深度学习 图像 目标检测 分割 跟踪 姿态估计 机器学习 神经网络 数据可视化 课堂实验 " * 6


def simple_tokenize(text: str) -> Counter:
    """用最简单的空格切分生成词频。"""
    words = [word for word in text.split() if word]
    return Counter(words)


def pick_available_font() -> str:
    """
    按顺序查找本机可用的中文字体。

    为什么不用一个写死路径：
    同样都是 Mac，不同版本系统里内置中文字体名称和目录可能不完全一样，
    按顺序探测更适合课堂机器和同学自己的电脑。
    """
    for font_path in FONT_CANDIDATES:
        if font_path.exists():
            return str(font_path)
    raise FileNotFoundError("没有找到可用的中文字体，请检查系统字体目录。")


def main():
    """生成并展示中文词云图，保存结果图片到脚本目录。"""
    text = build_demo_text()
    frequencies = simple_tokenize(text)
    font_path = pick_available_font()

    cloud = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        font_path=font_path,
    ).generate_from_frequencies(frequencies)

    plt.figure(figsize=(12, 6))
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("教学词云示例")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.show()
    print(f"词云图片已保存到: {OUTPUT_PATH}")
    print(f"本次使用的中文字体: {font_path}")


if __name__ == "__main__":
    main()
