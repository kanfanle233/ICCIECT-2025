# 迁移清单

本文档记录了论文代码项目的标准化迁移过程。

## 迁移概要

- **迁移日期**: 2026-06-11
- **迁移范围**: `/Users/davidfang/PyCharmMiscProject/minicondapythonProject1/论文代码`
- **总文件数**: 232 个文件
- **总大小**: 约 94 MB
- **重复文件组数**: 12 组（共 24 个重复文件）

---

## 一、目录结构创建

### 新建目录

✅ `src/gaokao_recommender/` - Python模块目录  
✅ `scripts/pipeline_steps/` - 流水线脚本目录  
✅ `notebooks/` - Jupyter Notebooks目录  
✅ `data/raw/` - 原始数据目录  
✅ `data/interim/` - 中间数据目录  
✅ `data/processed/` - 处理后数据目录  
✅ `data/private/` - 私有数据目录  
✅ `models/` - 模型目录（含子目录）  
✅ `reports/figures/` - 图表目录  
✅ `reports/paper/` - 论文目录  
✅ `reports/references/` - 参考资料目录  
✅ `docs/` - 文档目录  
✅ `archive/` - 归档目录  
✅ `metadata/` - 元数据目录  

---

## 二、文件迁移记录

### 1. 临时文件归档

**移动到 `archive/`:**

- ✅ `_tmp_cell_51.py`
- ✅ `_tmp_cell_73.py`
- ✅ `_tmp_cell_75.py`
- ✅ `_tmp_cell_76.py`
- ✅ `_tmp_cell_80.py`
- ✅ `_tmp_高考数据处理大修版本-checkpoint_code.py`
- ✅ `高考数据处理大修版本-checkpoint.ipynb.broken`
- ✅ `高考数据处理大修版本-checkpoint.ipynb.from_restore`
- ✅ `__pycache___高考数据处理大修版本_py拆分_优化版/`

**共 9 个文件/目录**

### 2. 数据文件迁移

**私有数据 (`data/private/`):**

- ✅ `1_2023高三二模学生成绩单.xls`
- ✅ `二模成绩.xlsx`
- ✅ `二模_MBTI_预测结果.xlsx`
- ✅ `二模_MBTI_预测结果_plus60.xlsx`
- ✅ `二模_MBTI_预测结果_plus60.txt`
- ✅ `二模_MBTI_预测结果_无姓名.xlsx`
- ✅ `学生投档线合并结果.xlsx`
- ✅ `改进后_最终志愿推荐结果.xlsx`

**共 8 个文件**

**原始数据 (`data/raw/`):**

- ✅ `2023上海专业分数线.txt`
- ✅ `2023上海专业分数线.xlsx`
- ✅ `2023年考生高考成绩分布表（上海市）.txt`
- ✅ `2023年考生高考成绩分布表（上海市）.xlsx`
- ✅ `combined_2023.csv`
- ✅ `subject_combo_to_mbti.csv`
- ✅ `重要-不要在网盘预览表格，下载到自己电脑上打开.txt`
- ✅ 7 个历史数据目录（上海_专业分数线_2017-2023/ 等）

**共 7 个文件 + 7 个目录**

**处理后数据 (`data/processed/`):**

- ✅ `2023上海专业分数线_clean.csv`
- ✅ `2023上海专业分数线_model_ready.csv`
- ✅ `2023上海专业分数线_with_PredictedMBTI.csv`
- ✅ `2023年考生高考成绩分布表_clean.csv`
- ✅ `上海一分一段_2023_clean.csv`
- ✅ `combo_based_recommendations.csv`
- ✅ `combo_based_recommendations_sklearn.csv`
- ✅ `recommendation_candidates.csv`
- ✅ `未映射MBTI的专业列表.csv`
- ✅ `subject_combo_to_mbti_clean.csv`

**共 10 个文件**

### 3. 模型文件迁移

**模型目录 (`models/`):**

- ✅ `best_model.pth` → `models/lstm_rank/`
- ✅ `lstm_total_score.h5` → `models/lstm_rank/`
- ✅ `interpretable_model.pth` → `models/interpretability/`
- ✅ `interpretable_model_complete.pth` → `models/interpretability/`
- ✅ `lstm_rank_opt.pth` → `models/lstm_rank_opt/`
- ✅ `lstm_rank.h5` → `models/lstm_rank/`
- ✅ `lstm_rank_opt.h5` → `models/lstm_rank_opt/`
- ✅ `transformer_rank.h5` → `models/transformer_rank/`
- ✅ `scaler_rank.pkl` → `models/lstm_rank/`
- ✅ `scaler_rank_opt.pkl` → `models/lstm_rank_opt/`
- ✅ `mlp_rank.pkl` → `models/mlp_rank/`
- ✅ `lstm_rank_model/` 目录
- ✅ `lstm_rank_opt/` 目录
- ✅ `mlp_rank_model/` 目录
- ✅ `transformer_rank_model/` 目录

**共 15 个模型文件 + 4 个目录**

### 4. 报告和图表迁移

**图表 (`reports/figures/`):**

- ✅ 72 个 PNG/JPG 图表文件

**论文 (`reports/paper/`):**

- ✅ 11 个 DOC/DOCX 文件

**参考资料 (`reports/references/`):**

- ✅ 5 个 PDF 文件

### 5. 流水线脚本迁移

**脚本 (`scripts/pipeline_steps/`):**

- ✅ `001_数据读取与基础可视化.py`
- ✅ `002_清洗标准化与核心表生成.py`
- ✅ `003_探索分析与建模前检查.py`
- ✅ `004_LSTM基线训练评估与可视化.py`
- ✅ `005_优化版LSTM训练_自动GPU_CPU.py`
- ✅ `006_交叉验证与时序切分评估.py`
- ✅ `007_SVD推荐系统_向量化优化.py`
- ✅ `008_可解释性分析_自动GPU_CPU.py`
- ✅ `009_完整推荐流程与可视化输出.py`
- ✅ `010_多模型基准对比.py`

**共 10 个脚本**

### 6. Jupyter Notebooks迁移

**Notebooks (`notebooks/`):**

- ✅ `高考数据处理.ipynb`
- ✅ `高考数据处理大修版本.ipynb`

**共 2 个文件**

---

## 三、新建文件

### 配置文件

- ✅ `.gitignore` - Git忽略规则
- ✅ `src/gaokao_recommender/__init__.py` - Python包初始化
- ✅ `src/gaokao_recommender/paths.py` - 统一路径配置
- ✅ `src/gaokao_recommender/path_adapter.py` - 路径适配器

### 工具脚本

- ✅ `scripts/run_pipeline.py` - 统一入口点
- ✅ `scripts/update_paths.py` - 路径更新工具
- ✅ `scripts/batch_update_scripts.py` - 批量更新工具

### 文档

- ✅ `docs/README.md` - 项目说明
- ✅ `docs/data_dictionary.md` - 数据字典
- ✅ `docs/migration_checklist.md` - 迁移清单（本文件）

### 元数据

- ✅ `metadata/file_manifest.csv` - 文件清单（含校验和）

---

## 四、路径配置更新

### 更新的脚本

所有 001-010 流水线脚本已更新，现在使用统一路径配置：

```python
# 旧代码（已替换）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "志愿填报辅助系统" / "上海高考录取数据17-23年"
OUTPUT_DIR = DATA_DIR / "用到的"

# 新代码
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from gaokao_recommender.paths import *

DATA_DIR = DATA_RAW_DIR
OUTPUT_DIR = DATA_PROCESSED_DIR
```

### 路径配置模块

**`src/gaokao_recommender/paths.py`** 定义了以下常量：

- `PROJECT_ROOT` - 项目根目录
- `DATA_RAW_DIR` - 原始数据目录
- `DATA_INTERIM_DIR` - 中间数据目录
- `DATA_PROCESSED_DIR` - 处理后数据目录
- `DATA_PRIVATE_DIR` - 私有数据目录
- `MODEL_DIR` - 模型根目录
- `MODEL_LSTM_RANK_DIR` - LSTM排名模型目录
- `MODEL_LSTM_RANK_OPT_DIR` - 优化LSTM模型目录
- `MODEL_MLP_RANK_DIR` - MLP排名模型目录
- `MODEL_TRANSFORMER_RANK_DIR` - Transformer模型目录
- `MODEL_INTERPRETABILITY_DIR` - 可解释性模型目录
- `FIGURE_DIR` - 图表目录
- `PAPER_DIR` - 论文目录
- `REFERENCES_DIR` - 参考资料目录
- `DOCS_DIR` - 文档目录
- `ARCHIVE_DIR` - 归档目录
- `METADATA_DIR` - 元数据目录

---

## 五、Git 配置

### .gitignore 规则

```
# Python
__pycache__/
*.py[cod]
*$py.class

# Jupyter
.ipynb_checkpoints/

# Temporary
*.tmp
~$*
~WRL*.tmp

# macOS
.DS_Store

# Windows
Thumbs.db

# Private data
data/private/
```

---

## 六、重复文件处理

### 已识别的重复文件组（12组）

1. `高考数据处理大修版本-checkpoint.ipynb.from_restore` 和 `志愿填报辅助系统/高考数据处理大修版本.ipynb`
2. `高考数据处理-checkpoint.ipynb` 和 `.ipynb_checkpoints/高考数据处理-checkpoint.ipynb`
3. `志愿填报辅助系统/daima.docx` 和 `.ipynb_checkpoints/daima.docx`
4. `2023年考生高考成绩分布表（上海市）.txt` 的两个副本
5. `2023年考生高考成绩分布表_clean.csv` 和 `上海一分一段_2023_clean.csv`
6. `combo_based_recommendations.csv` 和 `combo_based_recommendations_sklearn.csv`
7. `2023上海专业分数线.xlsx` 的两个副本
8. `subject_combo_to_mbti.csv` 和 `subject_combo_to_mbti_clean.csv`
9. `2023年考生高考成绩分布表（上海市）.xlsx` 的两个副本
10. `time_split_residuals.png` 的两个副本
11. `2023上海专业分数线.txt` 的两个副本
12. `subject_combo_to_mbti.txt` 的两个副本

### 处理建议

所有重复文件已记录在 `metadata/file_manifest.csv` 中，标记为 `is_duplicate: yes`。建议在确认可复现后清理。

---

## 七、待完成工作

### 短期（1周内）

- [ ] 验证所有脚本从新路径读取数据
- [ ] 运行完整流水线测试
- [ ] 确认核心数据行数正确
- [ ] 清理重复文件（确认后）

### 中期（1个月内）

- [ ] 补充 API 文档
- [ ] 添加单元测试
- [ ] 创建 CI/CD 流水线
- [ ] 优化路径配置（添加环境变量支持）

### 长期（3个月内）

- [ ] 重构脚本为可复用模块
- [ ] 添加数据验证工具
- [ ] 创建数据版本控制方案
- [ ] 完善错误处理和日志

---

## 八、验证清单

### 运行验证

```bash
# 1. 验证目录结构
python src/gaokao_recommender/paths.py

# 2. 验证脚本更新
python scripts/pipeline_steps/001_数据读取与基础可视化.py

# 3. 验证流水线入口
python scripts/run_pipeline.py --list

# 4. 验证数据完整性
wc -l data/processed/2023上海专业分数线_clean.csv
# 应输出 3525（含表头）

wc -l data/processed/recommendation_candidates.csv
# 应输出 70481（含表头）

# 5. 验证Git配置
git status
# data/private/ 应该不显示
```

### 数据完整性验证

- [ ] 专业 clean 表行数: 3524 行 ✅
- [ ] 推荐候选表行数: 70480 行 ✅
- [ ] 推荐结果表行数: 200 行 ✅
- [ ] 无数据损坏
- [ ] 编码正确（UTF-8 with BOM）

---

## 九、回滚方案

如果迁移后出现问题，可以按以下步骤回滚：

1. **恢复原始目录结构**:
   ```bash
   # 从版本控制恢复
   git checkout HEAD -- .
   ```

2. **恢复脚本路径**:
   ```bash
   # 手动还原 001-010 脚本的路径配置
   ```

3. **恢复数据文件**:
   ```bash
   # 从 data/raw/, data/processed/, data/private/ 恢复文件到原位置
   ```

---

## 十、联系人

- **负责人**: [待填写]
- **迁移日期**: 2026-06-11
- **最后更新**: 2026-06-11

---

## 附录：文件统计

| 类型 | 数量 | 大小 |
|------|------|------|
| Python 脚本 | 20 | ~200 KB |
| 数据文件 (CSV/TXT) | 107 | ~50 MB |
| 图表 (PNG/JPG) | 51 | ~20 MB |
| 模型文件 (H5/PTH/PKL) | 15 | ~15 MB |
| 文档 (DOC/DOCX) | 10 | ~5 MB |
| 参考资料 (PDF) | 5 | ~3 MB |
| 归档文件 | 17 | ~1 MB |
| **总计** | **232** | **~94 MB** |
