"""
教学示例：check plot

- 功能：演示 数据可视化 中与“check plot”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""

import importlib

# 常用 Python 数据分析 / 数据可视化库
libs = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "plotly",
    "bokeh",
    "pyecharts",
    "wordcloud",
    "missingno",
    "folium",
    "geopandas",
    "networkx",
    "squarify",
    "yellowbrick",
]

print("Python 常用数据分析与可视化库安装检查：")
print("-" * 50)

missing = []
for lib in libs:
    try:
        module = importlib.import_module(lib)
        version = getattr(module, "__version__", "已安装，但无法读取版本")
        print(f"✅ {lib:<12} 已安装，版本：{version}")
    except ImportError:
        print(f"❌ {lib:<12} 未安装")
        missing.append(lib)

print("-" * 50)
if missing:
    print("以下库未安装：")
    for lib in missing:
        print(f" - {lib}")
    print("\n你可以通过 pip 安装这些库，例如：")
    # geopandas 推荐用 conda 安装，这里单独说明
    other_missing = [lib for lib in missing if lib != "geopandas"]
    if other_missing:
        libs_str = " ".join(other_missing)
        print(f"pip install {libs_str}")
    if "geopandas" in missing:
        print("conda install -c conda-forge geopandas")
else:
    print("所有库均已安装")

print("检查完成")