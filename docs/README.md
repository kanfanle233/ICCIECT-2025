# 高考志愿填报推荐系统

基于 LSTM 和协同过滤的高考志愿填报辅助系统，用于为上海高考考生提供个性化志愿推荐。

## 项目结构

```
├── src/gaokao_recommender/    # 可复用 Python 模块
│   ├── paths.py              # 统一路径配置
│   └── ...
├── scripts/                   # 流水线脚本
│   ├── pipeline_steps/       # 001-010 阶段脚本
│   └── run_pipeline.py       # 统一入口点
├── notebooks/                 # Jupyter Notebooks
├── data/                      # 数据文件
│   ├── raw/                  # 原始数据（不可修改）
│   ├── interim/              # 中间结果
│   ├── processed/            # 最终处理数据
│   └── private/              # 私有数据（学生信息）
├── models/                    # 训练好的模型
│   ├── lstm_rank/
│   ├── lstm_rank_opt/
│   ├── mlp_rank/
│   ├── transformer_rank/
│   └── interpretability/
├── reports/                   # 报告和图表
│   ├── figures/
│   ├── paper/
│   └── references/
├── docs/                      # 文档
├── archive/                   # 归档和临时文件
└── metadata/                  # 元数据
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 或手动安装主要依赖
pip install pandas numpy matplotlib seaborn
pip install torch torchvision
pip install scikit-learn
pip install xgboost lightgbm
```

### 2. 运行流水线

```bash
# 查看所有可用阶段
python scripts/run_pipeline.py --list

# 运行单个阶段
python scripts/run_pipeline.py --stage prepare
python scripts/run_pipeline.py --stage eda
python scripts/run_pipeline.py --stage train

# 运行所有阶段
python scripts/run_pipeline.py --stage all
```

### 3. 流水线阶段

| 阶段 | 名称 | 描述 |
|------|------|------|
| `prepare` | 数据准备 | 数据读取、清洗、标准化 |
| `eda` | 探索性分析 | 数据探索与建模前检查 |
| `train` | 模型训练 | LSTM训练与优化 |
| `recommend` | 推荐系统 | SVD推荐系统与向量化优化 |
| `explain` | 可解释性分析 | 模型可解释性分析 |
| `benchmark` | 基准对比 | 多模型基准对比 |

## 数据说明

### 核心数据文件

- **2023上海专业分数线.txt**: 2023年上海高考各专业录取分数线
- **2023年考生高考成绩分布表（上海市）.txt**: 2023年上海高考成绩分布
- **subject_combo_to_mbti.txt**: 选科组合与MBTI类型对应关系
- **二模成绩.xlsx**: 学生二模成绩（私有数据）

### 数据质量注意

1. **2023上海专业分数线.txt**: 有 9814 行，其中 4907 行完全重复
2. **combined_2023.csv**: 混合粒度汇总表，缺失率约 34.9%
3. **2023年考生高考成绩分布表（上海市）.txt**: 文件名显示2023，但表头显示2022

详见 [数据字典](data_dictionary.md) 和 [迁移清单](migration_checklist.md)

## 路径配置

所有路径通过 `src/gaokao_recommender/paths.py` 统一管理：

```python
from gaokao_recommender.paths import *

# 使用示例
DATA_DIR = DATA_RAW_DIR
OUTPUT_DIR = DATA_PROCESSED_DIR
MODEL_DIR = MODEL_LSTM_RANK_DIR
FIGURE_DIR = FIGURE_DIR
```

## 重要提示

- **不要修改 `data/raw/` 中的文件**：这些是原始数据，应保持原样
- **私有数据已排除**：`data/private/` 中的文件包含学生个人信息，已加入 `.gitignore`
- **脚本输出位置**：所有脚本输出写入 `data/processed/`、`reports/figures/`、`models/` 等目录
- **不要在原始数据目录写入**：旧脚本的硬编码路径已更新，不再向原始目录写入

## 文档

- [数据字典](data_dictionary.md) - 详细的数据文件说明
- [迁移清单](migration_checklist.md) - 文件迁移记录
- [API文档](api.md) - Python模块API（待补充）

## 开发指南

### 添加新脚本

1. 在 `scripts/pipeline_steps/` 中创建新脚本
2. 使用统一的路径配置：
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
   from gaokao_recommender.paths import *
   ```
3. 输出文件写入相应的标准化目录

### 修改路径配置

修改 `src/gaokao_recommender/paths.py` 中的常量，所有脚本会自动使用新路径。

## 许可证

[待添加]

## 联系方式

[待添加]
