# Intelligent Academic Planning Decision System Based on Multi-Dimensional Data Fusion and LSTM-SVD Collaboration

[English](README.md) | [中文](README.zh-CN.md)

<p align="center">
  <strong>ICCIECT 2025</strong><br>
  <em>International Conference on Computer Information and Education Engineering Technology</em>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#pipeline">Pipeline</a> •
  <a href="#results">Results</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/MAE-0.024-green" alt="MAE">
  <img src="https://img.shields.io/badge/R²-0.92-blue" alt="R²">
  <img src="https://img.shields.io/badge/Top--N-10-orange" alt="Top-N">
  <img src="https://img.shields.io/badge/Python-3.9+-yellow" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/Device-CUDA%20%7C%20MPS%20%7C%20CPU-brightgreen" alt="Device">
</p>

---

## Overview

This repository presents an intelligent recommendation system for **Shanghai Gaokao (college entrance exam) students** to make personalized university major choices. The system fuses multi-dimensional data including historical admission records, score distributions, MBTI personality mappings, and collaborative filtering to generate optimized major recommendations.

**Key Innovation:** Combining LSTM neural networks for rank prediction with SVD-based collaborative filtering and MBTI personality analysis to create a holistic academic planning system.

### Research Highlights

- **LSTM Rank Prediction**: MAE = 0.024, R² = 0.92 (10× improvement over linear regression baseline)
- **SVD Collaborative Filtering**: 48-dimensional embeddings with BPR optimization and MMR reranking
- **MBTI Personality Mapping**: Automatic mapping from Shanghai's "6-choose-3" subject combinations to personality types
- **Model Interpretability**: SHAP (Tree SHAP) analysis for transparent decision-making

---

## Features

### Core Capabilities

- **Multi-Dimensional Data Fusion**: Integrates 7+ data sources including admission scores, university rankings, and student preferences
- **LSTM Neural Network**: Advanced time-series modeling for score-to-rank prediction
- **Multi-Device Support**: Automatic detection and optimization for **CUDA**, **MPS (Apple Silicon)**, and **CPU**
- **SVD Collaborative Filtering**: Discovers latent interest-major associations through matrix factorization
- **MBTI Personality Integration**: Maps elective subject choices to personality types for better matching
- **MMR Reranking**: Ensures recommendation diversity with Maximal Marginal Relevance (λ=0.85)
- **SHAP Interpretability**: Provides transparent model explanations with Tree SHAP analysis
- **Cross-Platform Compatibility**: Works seamlessly on Windows, macOS, and Linux

### Technical Stack

- **Deep Learning**: PyTorch (LSTM, MLP, Transformer)
- **Machine Learning**: scikit-learn (SVD, Random Forest, GBDT, XGBoost)
- **Device Optimization**: CUDA, MPS (Apple Silicon GPU), CPU auto-detection
- **Recommendation**: Custom BPR (Bayesian Personalized Ranking) optimization
- **Interpretability**: SHAP (SHapley Additive exPlanations)
- **Data Processing**: pandas, NumPy
- **Visualization**: matplotlib, seaborn
- **Cross-Platform**: Windows, macOS, Linux support

---

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Sources (7+)                         │
│  • Historical Admission Records (2017-2023)                │
│  • Score Distributions & Rank Tables                        │
│  • University Rankings (QS, US News, etc.)                 │
│  • Subject Combinations & MBTI Mappings                    │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Processing Pipeline (001-003)             │
│  • Data Ingestion & Cleaning                                │
│  • Feature Engineering                                      │
│  • Exploratory Analysis                                     │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Training & Evaluation (004-006)          │
│  • LSTM Baseline Training (GPU/CPU auto-detect)            │
│  • Cross-Validation & Time-Series Split                    │
│  • Model Optimization                                       │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              Recommendation Engine (007)                     │
│  • TruncatedSVD (48-dim embeddings)                        │
│  • BPR Pairwise Loss Optimization                          │
│  • MMR Reranking (λ=0.85)                                  │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              Interpretability & Output (008-009)            │
│  • SHAP TreeExplainer Analysis                             │
│  • Gradient-based Feature Importance                       │
│  • Per-User Top-10 Recommendations Visualization           │
└─────────────────────────────────────────────────────────────┘
```

### Core Algorithms

#### 1. LSTM Rank Prediction
- **Architecture**: Multi-layer LSTM with dropout and batch normalization
- **Input**: Historical score sequences, percentile distributions
- **Output**: Predicted rank percentile
- **Performance**: MAE = 0.024, R² = 0.92

#### 2. SVD Collaborative Filtering
- **Method**: TruncatedSVD (48 dimensions) + BPR pairwise loss
- **Optimization**: Stochastic Gradient Descent (35 epochs)
- **Reranking**: MMR (Maximal Marginal Relevance) for diversity
- **Rating Formula**: `rating = mbti_w × (1 - difficulty) × 4 + 1`

#### 3. MBTI Personality Mapping
- **Input**: Shanghai "6-choose-3" subject combinations
- **Output**: MBTI personality type (e.g., INTJ, ENFP)
- **Matching**: Exact match (weight=1.0) or prefix match (weight=0.5)

---

## Project Structure

```
├── scripts/
│   ├── run_pipeline.py                 # Pipeline orchestrator
│   └── pipeline_steps/
│       ├── 001_数据读取与基础可视化.py     # Data ingestion
│       ├── 002_清洗标准化与核心表生成.py     # Data cleaning
│       ├── 003_探索分析与建模前检查.py           # EDA
│       ├── 004_LSTM基线训练评估与可视化.py      # LSTM baseline
│       ├── 005_优化版LSTM训练_自动GPU_CPU.py        # Optimized LSTM
│       ├── 006_交叉验证与时序切分评估.py             # Cross-validation
│       ├── 007_SVD推荐系统_向量化优化.py          # SVD recommendation
│       ├── 008_可解释性分析_自动GPU_CPU.py         # SHAP analysis
│       ├── 009_完整推荐流程与可视化输出.py           # End-to-end pipeline
│       └── 010_多模型基准对比.py       # Multi-model benchmark
├── data/
│   ├── raw/                            # 106 raw data files
│   ├── processed/                      # 10 cleaned datasets
│   └── private/                        # Student data (gitignored)
├── models/
│   ├── lstm_*.pth                      # LSTM model weights
│   ├── mlp_*.pth                       # MLP model weights
│   └── transformer_*.pth              # Transformer model weights
├── reports/
│   ├── figures/                        # 40+ visualizations
│   ├── paper/                          # Paper documents (CN/EN)
│   └── references/                     # Reference papers
├── src/
│   └── gaokao_recommender/
│       ├── paths.py                    # Path configuration
│       └── device_utils.py             # Device detection (CUDA/MPS/CPU)
├── notebooks/                          # Jupyter notebooks
└── docs/                               # Documentation
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (optional, for NVIDIA GPU acceleration)
- MPS support (for Apple Silicon Macs)
- 8GB+ RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/kanfanle233/ICCIECT-2025.git
cd ICCIECT-2025

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio
pip install scikit-learn pandas numpy matplotlib seaborn
pip install shap xgboost
```

### Device Support

The system automatically detects and optimizes for your hardware:

```python
from gaokao_recommender.device_utils import get_device

# Automatic device detection
device, device_type = get_device(verbose=True)

# Output examples:
# ✅ 检测到 CUDA GPU (NVIDIA)
# ✅ 检测到 MPS (Apple Silicon GPU)
# ⚠️ 使用 CPU
```

### Run Pipeline

```bash
# Execute the full pipeline (10 stages)
python scripts/run_pipeline.py

# Or run individual stages
python scripts/pipeline_steps/007_SVD推荐系统.py  # SVD recommendation
python scripts/pipeline_steps/008_可解释性分析.py  # SHAP analysis
```

### Generate Recommendations

```python
from src.gaokao_recommender import paths
import pandas as pd

# Load processed data
candidates = pd.read_csv(paths.processed / "recommendation_candidates.csv")

# Generate recommendations for a student
# (See scripts/pipeline_steps/009_推荐流水线.py for full implementation)
```

---

## Pipeline Stages

### Stage 001: Data Ingestion
- Read raw data files (admission scores, university rankings)
- Visualize MBTI distribution and score lines
- Output: Processed data files

### Stage 002: Data Cleaning
- Clean and normalize raw data
- Generate core tables for modeling
- Output: Standardized datasets

### Stage 003: Exploratory Analysis
- Statistical analysis and visualization
- Pre-modeling checks and data quality assessment
- Output: Analysis reports and figures

### Stage 004: LSTM Baseline
- Train baseline LSTM model
- Evaluate performance (MAE, R²)
- Output: Model weights and metrics

### Stage 005: Optimized LSTM
- **Multi-Device Support**: Automatic detection for CUDA, MPS (Apple Silicon), and CPU
- Hyperparameter tuning with device-specific optimization
- Adaptive batch size based on device capabilities
- Output: Optimized model weights

### Stage 006: Cross-Validation
- K-fold cross-validation
- Time-series split evaluation
- Output: Validation metrics

### Stage 007: SVD Recommendation
- TruncatedSVD embedding (48 dimensions)
- BPR pairwise loss optimization (35 epochs)
- MMR reranking (λ=0.85)
- Output: Recommendation model

### Stage 008: Interpretability
- SHAP TreeExplainer analysis
- Gradient-based feature importance
- Per-sample case analysis
- Output: Interpretability reports

### Stage 009: End-to-End Pipeline
- Full recommendation flow
- Per-user Top-10 visualization
- Output: Final recommendations

### Stage 010: Multi-Model Benchmark
- Compare LSTM, MLP, Transformer, GBDT, XGBoost
- Performance benchmarking
- Output: Comparison charts

---

## Results

### Key Metrics

| Model | MAE | R² | Training Time |
|-------|-----|----|---------------|
| **LSTM (Ours)** | **0.024** | **0.92** | 45 min |
| MLP | 0.031 | 0.88 | 20 min |
| Transformer | 0.028 | 0.90 | 60 min |
| GBDT | 0.035 | 0.85 | 5 min |
| XGBoost | 0.033 | 0.86 | 8 min |
| Linear Regression | 0.24 | 0.12 | 0.5 min |

### Visualizations

<p align="center">
  <em>See <a href="reports/figures/">reports/figures/</a> for 40+ visualizations including:</em>
</p>

- LSTM training loss curves
- SHAP feature importance plots
- MBTI personality distribution
- Top-10 recommendation examples
- Model comparison charts

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{fang2025intelligent,
  title={Intelligent Academic Planning Decision System Based on Multi-Dimensional Data Fusion and LSTM-SVD Collaboration},
  author={Fang, Xinzhe and Yuan, Yue},
  booktitle={Proceedings of the International Conference on Computer Information and Education Engineering Technology (ICCIECT)},
  year={2025},
  organization={IEEE}
}
```

---

## Data Sources

- **Shanghai Admission Data (2017-2023)**: Historical admission scores, enrollment plans, and score distributions
- **University Rankings**: QS World University Rankings, US News Rankings, ShanghaiRanking
- **MBTI Mappings**: Subject combination to personality type mappings

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Shanghai Education Examination Authority for providing historical admission data
- Kaggle and educational data platforms for supplementary datasets
- The open-source community for excellent ML frameworks

---

<p align="center">
  <strong>ICCIECT 2025</strong><br>
  <em>Advancing Education Through Intelligent Systems</em>
</p>

---

**Contact**: Xinzhe Fang - [GitHub](https://github.com/kanfanle233)

**Project Link**: [https://github.com/kanfanle233/ICCIECT-2025](https://github.com/kanfanle233/ICCIECT-2025)
