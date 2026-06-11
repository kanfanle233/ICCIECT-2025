# 基于多维数据融合与LSTM-SVD协同的学业规划智能决策系统

[English](README.md) | [中文](README.zh-CN.md)

<p align="center">
  <strong>ICCIECT 2025</strong><br>
  <em>国际计算机信息与教育工程技术会议</em>
</p>

<p align="center">
  <a href="#功能特性">功能特性</a> •
  <a href="#技术架构">技术架构</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#流水线阶段">流水线阶段</a> •
  <a href="#实验结果">实验结果</a> •
  <a href="#论文引用">论文引用</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/MAE-0.024-green" alt="MAE">
  <img src="https://img.shields.io/badge/R²-0.92-blue" alt="R²">
  <img src="https://img.shields.io/badge/Top--N-10-orange" alt="Top-N">
  <img src="https://img.shields.io/badge/Python-3.9+-yellow" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/设备-CUDA%20%7C%20MPS%20%7C%20CPU-brightgreen" alt="设备">
</p>

---

## 项目概述

本项目为**上海高考考生**提供个性化大学专业推荐系统，融合历史录取数据、分数分布、MBTI人格映射和协同过滤等多维度数据，生成优化的专业推荐方案。

**核心创新**：将LSTM神经网络用于排名预测，结合SVD协同过滤和MBTI人格分析，构建全面的学业规划决策系统。

### 研究亮点

- **LSTM排名预测**：MAE = 0.024，R² = 0.92（比线性回归基线提升10倍）
- **SVD协同过滤**：48维嵌入 + BPR优化 + MMR重排序
- **MBTI人格映射**：自动将上海"6选3"选科组合映射到MBTI人格类型
- **模型可解释性**：SHAP (Tree SHAP) 分析实现透明决策

---

## 功能特性

### 核心能力

- **多维数据融合**：整合7+数据源，包括录取分数、大学排名和学生偏好
- **LSTM神经网络**：先进的时间序列建模，用于分数到排名的预测
- **多设备支持**：自动检测并优化 **CUDA**、**MPS (Apple Silicon GPU)** 和 **CPU**
- **SVD协同过滤**：通过矩阵分解挖掘潜在的兴趣-专业关联
- **MBTI人格整合**：将选科选择映射到人格类型，实现更精准匹配
- **MMR重排序**：使用最大边际相关性(λ=0.85)确保推荐多样性
- **SHAP可解释性**：提供透明的模型解释，使用Tree SHAP分析
- **跨平台兼容性**：支持 Windows、macOS 和 Linux 系统

### 技术栈

- **深度学习**：PyTorch (LSTM, MLP, Transformer)
- **机器学习**：scikit-learn (SVD, Random Forest, GBDT, XGBoost)
- **设备优化**：CUDA、MPS (Apple Silicon GPU)、CPU 自动检测
- **推荐系统**：自定义BPR（贝叶斯个性化排序）优化
- **可解释性**：SHAP (SHapley Additive exPlanations)
- **数据处理**：pandas, NumPy
- **可视化**：matplotlib, seaborn
- **跨平台支持**：Windows、macOS、Linux

---

## 技术架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    数据源 (7+)                               │
│  • 历史录取数据 (2017-2023)                                │
│  • 分数分布与排名表                                         │
│  • 大学排名 (QS, US News等)                                │
│  • 选科组合与MBTI映射                                       │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              数据处理流水线 (001-003)                        │
│  • 数据导入与清洗                                           │
│  • 特征工程                                                 │
│  • 探索性分析                                               │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              模型训练与评估 (004-006)                        │
│  • LSTM基线训练 (GPU/CPU自动检测)                          │
│  • 交叉验证与时序切分                                       │
│  • 模型优化                                                 │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              推荐引擎 (007)                                  │
│  • TruncatedSVD (48维嵌入)                                 │
│  • BPR成对损失优化                                          │
│  • MMR重排序 (λ=0.85)                                      │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              可解释性与输出 (008-009)                        │
│  • SHAP TreeExplainer分析                                   │
│  • 梯度特征重要性                                           │
│  • 用户Top-10推荐可视化                                     │
└─────────────────────────────────────────────────────────────┘
```

### 核心算法

#### 1. LSTM排名预测
- **架构**：多层LSTM + Dropout + 批归一化
- **输入**：历史分数序列、百分位分布
- **输出**：预测排名百分位
- **性能**：MAE = 0.024，R² = 0.92

#### 2. SVD协同过滤
- **方法**：TruncatedSVD (48维) + BPR成对损失
- **优化**：随机梯度下降 (35轮)
- **重排序**：MMR（最大边际相关性）确保多样性
- **评分公式**：`rating = mbti_w × (1 - difficulty) × 4 + 1`

#### 3. MBTI人格映射
- **输入**：上海"6选3"选科组合
- **输出**：MBTI人格类型 (如INTJ, ENFP)
- **匹配**：精确匹配 (权重=1.0) 或前缀匹配 (权重=0.5)

---

## 项目结构

```
├── scripts/
│   ├── run_pipeline.py                 # 流水线编排器
│   └── pipeline_steps/
│       ├── 001_数据读取与基础可视化.py     # 数据导入
│       ├── 002_清洗标准化与核心表生成.py     # 数据清洗
│       ├── 003_探索分析与建模前检查.py           # 探索性分析
│       ├── 004_LSTM基线训练评估与可视化.py      # LSTM基线
│       ├── 005_优化版LSTM训练_自动GPU_CPU.py        # 优化LSTM
│       ├── 006_交叉验证与时序切分评估.py             # 交叉验证
│       ├── 007_SVD推荐系统_向量化优化.py          # SVD推荐
│       ├── 008_可解释性分析_自动GPU_CPU.py         # SHAP分析
│       ├── 009_完整推荐流程与可视化输出.py           # 端到端流水线
│       └── 010_多模型基准对比.py       # 多模型对比
├── data/
│   ├── raw/                            # 106个原始数据文件
│   ├── processed/                      # 10个清洗后数据集
│   └── private/                        # 学生数据 (gitignore)
├── models/
│   ├── lstm_*.pth                      # LSTM模型权重
│   ├── mlp_*.pth                       # MLP模型权重
│   └── transformer_*.pth              # Transformer模型权重
├── reports/
│   ├── figures/                        # 40+可视化图表
│   ├── paper/                          # 论文文档 (中英文)
│   └── references/                     # 参考文献
├── src/
│   └── gaokao_recommender/
│       ├── paths.py                    # 路径配置
│       └── device_utils.py             # 设备检测 (CUDA/MPS/CPU)
├── notebooks/                          # Jupyter笔记本
└── docs/                               # 文档
```

---

## 快速开始

### 环境要求

- Python 3.9+
- CUDA 11.8+ (可选，用于 NVIDIA GPU 加速)
- MPS 支持 (用于 Apple Silicon Mac)
- 8GB+ RAM

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/kanfanle233/ICCIECT-2025.git
cd ICCIECT-2025

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install torch torchvision torchaudio
pip install scikit-learn pandas numpy matplotlib seaborn
pip install shap xgboost
```

### 设备支持

系统自动检测并优化您的硬件配置：

```python
from gaokao_recommender.device_utils import get_device

# 自动设备检测
device, device_type = get_device(verbose=True)

# 输出示例：
# ✅ 检测到 CUDA GPU (NVIDIA)
# ✅ 检测到 MPS (Apple Silicon GPU)
# ⚠️ 使用 CPU
```

### 运行流水线

```bash
# 执行完整流水线 (10个阶段)
python scripts/run_pipeline.py

# 或单独运行某个阶段
python scripts/pipeline_steps/007_SVD推荐系统.py  # SVD推荐
python scripts/pipeline_steps/008_可解释性分析.py  # SHAP分析
```

### 生成推荐

```python
from src.gaokao_recommender import paths
import pandas as pd

# 加载处理后的数据
candidates = pd.read_csv(paths.processed / "recommendation_candidates.csv")

# 为学生生成推荐
# (完整实现见 scripts/pipeline_steps/009_推荐流水线.py)
```

---

## 流水线阶段

### 阶段001：数据导入
- 读取原始数据文件（录取分数、大学排名）
- 可视化MBTI分布和分数曲线
- 输出：处理后的数据文件

### 阶段002：数据清洗
- 清洗和标准化原始数据
- 生成建模核心表格
- 输出：标准化数据集

### 阶段003：探索性分析
- 统计分析和可视化
- 建模前检查和数据质量评估
- 输出：分析报告和图表

### 阶段004：LSTM基线
- 训练基线LSTM模型
- 评估性能 (MAE, R²)
- 输出：模型权重和指标

### 阶段005：优化LSTM
- **多设备支持**：自动检测 CUDA、MPS (Apple Silicon) 和 CPU
- 根据设备特性进行超参数优化
- 自适应 batch size 以充分利用硬件性能
- 输出：优化后的模型权重

### 阶段006：交叉验证
- K折交叉验证
- 时序切分评估
- 输出：验证指标

### 阶段007：SVD推荐
- TruncatedSVD嵌入 (48维)
- BPR成对损失优化 (35轮)
- MMR重排序 (λ=0.85)
- 输出：推荐模型

### 阶段008：可解释性
- SHAP TreeExplainer分析
- 梯度特征重要性
- 单样本案例分析
- 输出：可解释性报告

### 阶段009：端到端流水线
- 完整推荐流程
- 用户Top-10可视化
- 输出：最终推荐结果

### 阶段10：多模型对比
- 对比LSTM, MLP, Transformer, GBDT, XGBoost
- 性能基准测试
- 输出：对比图表

---

## 实验结果

### 关键指标

| 模型 | MAE | R² | 训练时间 |
|------|-----|----|----------  |
| **LSTM (我们的)** | **0.024** | **0.92** | 45分钟 |
| MLP | 0.031 | 0.88 | 20分钟 |
| Transformer | 0.028 | 0.90 | 60分钟 |
| GBDT | 0.035 | 0.85 | 5分钟 |
| XGBoost | 0.033 | 0.86 | 8分钟 |
| 线性回归 | 0.24 | 0.12 | 0.5分钟 |

### 可视化图表

<p align="center">
  <em>参见 <a href="reports/figures/">reports/figures/</a> 获取40+可视化图表，包括：</em>
</p>

- LSTM训练损失曲线
- SHAP特征重要性图
- MBTI人格分布图
- Top-10推荐示例
- 模型对比图表

---

## 论文引用

如果您在研究中使用了本代码，请引用：

```bibtex
@inproceedings{fang2025intelligent,
  title={基于多维数据融合与LSTM-SVD协同的学业规划智能决策系统},
  author={方新哲 and 袁月},
  booktitle={国际计算机信息与教育工程技术会议 (ICCIECT)},
  year={2025},
  organization={IEEE}
}
```

---

## 数据来源

- **上海录取数据 (2017-2023)**：历史录取分数、招生计划和分数分布
- **大学排名**：QS世界大学排名、US News排名、上海软科排名
- **MBTI映射**：选科组合到人格类型的映射

---

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 致谢

- 上海市教育考试院提供的历史录取数据
- Kaggle和教育数据平台的补充数据集
- 开源社区提供的优秀机器学习框架

---

<p align="center">
  <strong>ICCIECT 2025</strong><br>
  <em>通过智能系统推动教育发展</em>
</p>

---

**联系人**：方新哲 - [GitHub](https://github.com/kanfanle233)

**项目链接**：[https://github.com/kanfanle233/ICCIECT-2025](https://github.com/kanfanle233/ICCIECT-2025)
