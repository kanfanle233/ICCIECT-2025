"""
统一路径配置模块
================

定义所有目录路径常量，确保整个项目使用一致的路径。
"""

from pathlib import Path

# 项目根目录 (相对于此文件)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# 数据目录
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_PRIVATE_DIR = PROJECT_ROOT / "data" / "private"

# 模型目录
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_LSTM_RANK_DIR = MODEL_DIR / "lstm_rank"
MODEL_LSTM_RANK_OPT_DIR = MODEL_DIR / "lstm_rank_opt"
MODEL_MLP_RANK_DIR = MODEL_DIR / "mlp_rank"
MODEL_TRANSFORMER_RANK_DIR = MODEL_DIR / "transformer_rank"
MODEL_INTERPRETABILITY_DIR = MODEL_DIR / "interpretability"

# 报告目录
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
PAPER_DIR = PROJECT_ROOT / "reports" / "paper"
REFERENCES_DIR = PROJECT_ROOT / "reports" / "references"

# 文档目录
DOCS_DIR = PROJECT_ROOT / "docs"

# 归档目录
ARCHIVE_DIR = PROJECT_ROOT / "archive"

# 元数据目录
METADATA_DIR = PROJECT_ROOT / "metadata"

# 脚本目录
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PIPELINE_STEPS_DIR = SCRIPTS_DIR / "pipeline_steps"

# 确保所有必要目录存在
def ensure_directories():
    """创建所有必要的目录（如果不存在）"""
    directories = [
        DATA_RAW_DIR,
        DATA_INTERIM_DIR,
        DATA_PROCESSED_DIR,
        DATA_PRIVATE_DIR,
        MODEL_DIR,
        MODEL_LSTM_RANK_DIR,
        MODEL_LSTM_RANK_OPT_DIR,
        MODEL_MLP_RANK_DIR,
        MODEL_TRANSFORMER_RANK_DIR,
        MODEL_INTERPRETABILITY_DIR,
        FIGURE_DIR,
        PAPER_DIR,
        REFERENCES_DIR,
        DOCS_DIR,
        ARCHIVE_DIR,
        METADATA_DIR,
        SCRIPTS_DIR,
        PIPELINE_STEPS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ All directories ensured at: {PROJECT_ROOT}")

if __name__ == "__main__":
    ensure_directories()
