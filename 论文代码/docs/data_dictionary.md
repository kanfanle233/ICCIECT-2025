# 数据字典

本文档描述项目中使用的所有数据文件。

## 目录

1. [原始数据 (data/raw/)](#原始数据)
2. [处理后数据 (data/processed/)](#处理后数据)
3. [私有数据 (data/private/)](#私有数据)
4. [历史数据](#历史数据)

---

## 原始数据

### 2023上海专业分数线.txt
- **位置**: `data/raw/2023上海专业分数线.txt`
- **格式**: TXT (Tab分隔)
- **编码**: UTF-8 with BOM
- **行数**: 9814 行（其中 4907 行完全重复）
- **实际有效行数**: 4907 行（去重后）
- **列**:
  - `年份`: 录取年份（2017-2023）
  - `院校代码`: 高校代码
  - `院校名称`: 高校名称
  - `专业代码`: 专业代码
  - `专业名称`: 专业名称
  - `选科要求`: 选科要求
  - `最低分`: 录取最低分
  - `最低位次`: 录取最低位次
- **注意**: 包含大量重复行，需要在使用前去重

### 2023年考生高考成绩分布表（上海市）.txt
- **位置**: `data/raw/2023年考生高考成绩分布表（上海市）.txt`
- **格式**: TXT (Tab分隔)
- **行数**: 约 300 行
- **⚠️ 注意**: 文件名显示2023，但表头显示2022。需人工确认数据来源年份
- **列**:
  - `分数段`: 分数区间
  - `人数`: 该分数段人数
  - `累计人数`: 累计人数
  - `百分位`: 百分位排名

### subject_combo_to_mbti.txt
- **位置**: `data/raw/subject_combo_to_mbti.txt`
- **格式**: TXT (Tab分隔)
- **行数**: 约 20 行
- **描述**: 选科组合与MBTI类型对应关系表
- **列**:
  - `subject_combo`: 选科组合（如 "物化生"）
  - `mbti`: MBTI类型（如 "INTJ"）
  - `count`: 出现次数

### combined_2023.csv
- **位置**: `data/raw/combined_2023.csv`
- **格式**: CSV
- **行数**: 10176 行
- **重复行数**: 4907 行
- **平均缺失率**: 约 34.9%
- **描述**: 混合粒度汇总表，包含多个来源的数据
- **⚠️ 注意**: 不能作为核心唯一事实表，需要根据业务需求选择性使用

### 上海高考录取数据17-23年/
- **位置**: `data/raw/上海高考录取数据17-23年/`
- **描述**: 2017-2023年上海高考录取历史数据目录
- **子目录**:
  - `上海_专业分数线_2017-2023/`: 历年专业分数线
  - `上海_投档线_2017-2023/`: 历年投档线
  - `上海_招生计划_2017-2023/`: 历年招生计划
  - `上海_最新资料_2023/`: 2023年最新资料
  - `上海_其他资料/`: 其他参考资料
  - `全国通用高考数据/`: 全国通用数据
  - `用到的/`: 实际使用的数据副本

---

## 处理后数据

### 2023上海专业分数线_clean.csv
- **位置**: `data/processed/2023上海专业分数线_clean.csv`
- **格式**: CSV
- **行数**: 3524 行（无重复）
- **描述**: 清洗后的专业分数线数据，已去重
- **列**: 同原始数据

### 2023上海专业分数线_model_ready.csv
- **位置**: `data/processed/2023上海专业分数线_model_ready.csv`
- **格式**: CSV
- **行数**: 约 3000 行
- **描述**: 模型输入格式，已标准化特征

### 2023上海专业分数线_with_PredictedMBTI.csv
- **位置**: `data/processed/2023上海专业分数线_with_PredictedMBTI.csv`
- **格式**: CSV
- **描述**: 包含预测MBTI类型的专业分数线数据

### 2023年考生高考成绩分布表_clean.csv
- **位置**: `data/processed/2023年考生高考成绩分布表_clean.csv`
- **格式**: CSV
- **行数**: 约 300 行
- **描述**: 清洗后的成绩分布表

### 上海一分一段_2023_clean.csv
- **位置**: `data/processed/上海一分一段_2023_clean.csv`
- **格式**: CSV
- **行数**: 约 300 行
- **描述**: 2023年上海高考一分一段表（去重）

### recommendation_candidates.csv
- **位置**: `data/processed/recommendation_candidates.csv`
- **格式**: CSV
- **行数**: 70480 行
- **描述**: 推荐候选表，包含所有可能的专业-学生组合

### combo_based_recommendations.csv
- **位置**: `data/processed/combo_based_recommendations.csv`
- **格式**: CSV
- **行数**: 200 行
- **描述**: 基于选科组合的推荐结果

### combo_based_recommendations_sklearn.csv
- **位置**: `data/processed/combo_based_recommendations_sklearn.csv`
- **格式**: CSV
- **行数**: 200 行
- **描述**: 使用sklearn的推荐结果

### 未映射MBTI的专业列表.csv
- **位置**: `data/processed/未映射MBTI的专业列表.csv`
- **格式**: CSV
- **描述**: 未能映射MBTI类型的专业列表

### subject_combo_to_mbti_clean.csv
- **位置**: `data/processed/subject_combo_to_mbti_clean.csv`
- **格式**: CSV
- **描述**: 清洗后的选科-MBTI对应关系表

---

## 私有数据

### 1_2023高三二模学生成绩单.xls
- **位置**: `data/private/1_2023高三二模学生成绩单.xls`
- **格式**: Excel
- **⚠️ 敏感性**: 包含学生姓名、考号、班级等个人信息
- **处理建议**: 仅用于本地测试，不提交到版本控制

### 二模成绩.xlsx
- **位置**: `data/private/二模成绩.xlsx`
- **格式**: Excel
- **⚠️ 敏感性**: 同上

### 二模_MBTI_预测结果.xlsx
- **位置**: `data/private/二模_MBTI_预测结果.xlsx`
- **格式**: Excel
- **描述**: 包含学生MBTI预测结果

### 二模_MBTI_预测结果_plus60.xlsx
- **位置**: `data/private/二模_MBTI_预测结果_plus60.xlsx`
- **格式**: Excel
- **描述**: 扩展版MBTI预测结果

### 二模_MBTI_预测结果_plus60.txt
- **位置**: `data/private/二模_MBTI_预测结果_plus60.txt`
- **格式**: TXT
- **描述**: 文本格式的MBTI预测结果

### 二模_MBTI_预测结果_无姓名.xlsx
- **位置**: `data/private/二模_MBTI_预测结果_无姓名.xlsx`
- **格式**: Excel
- **描述**: 脱敏后的MBTI预测结果（已去除姓名）

### 学生投档线合并结果.xlsx
- **位置**: `data/private/学生投档线合并结果.xlsx`
- **格式**: Excel
- **描述**: 学生投档线合并数据

### 改进后_最终志愿推荐结果.xlsx
- **位置**: `data/private/改进后_最终志愿推荐结果.xlsx`
- **格式**: Excel
- **描述**: 改进后的最终推荐结果

---

## 历史数据

### 上海_专业分数线_2017-2023/
- **位置**: `data/raw/上海_专业分数线_2017-2023/`
- **格式**: Excel (每年一个文件)
- **文件列表**:
  - 2017上海专业分数线.xlsx
  - 2018上海专业分数线.xlsx
  - 2019上海专业分数线.xlsx
  - 2020上海专业分数线.xlsx
  - 2021上海专业分数线.xlsx
  - 2022上海专业分数线.xlsx
  - 2023上海专业分数线.xlsx

### 上海_投档线_2017-2023/
- **位置**: `data/raw/上海_投档线_2017-2023/`
- **格式**: Excel
- **描述**: 2017-2023年上海高考投档线数据

### 上海_招生计划_2017-2023/
- **位置**: `data/raw/上海_招生计划_2017-2023/`
- **格式**: Excel
- **描述**: 2017-2023年招生计划数据

---

## 数据验证

### 核心数据验证命令

```bash
# 验证专业clean表
wc -l data/processed/2023上海专业分数线_clean.csv
# 应输出 3524 行（或 3525 包括表头）

# 验证推荐候选表
wc -l data/processed/recommendation_candidates.csv
# 应输出 70480 行（或 70481 包括表头）

# 验证推荐结果表
wc -l data/processed/combo_based_recommendations.csv
# 应输出 200 行（或 201 包括表头）
```

### 数据完整性检查

```bash
# 检查是否有重复文件
python scripts/check_duplicates.py

# 检查文件完整性
python scripts/verify_data_integrity.py
```

---

## 注意事项

1. **不要修改原始数据**: `data/raw/` 中的文件应保持原样
2. **敏感数据保护**: `data/private/` 中的文件已加入 `.gitignore`
3. **数据来源确认**: 部分数据年份需要人工确认（见上文警告）
4. **重复文件处理**: 12组重复文件已标记在 `metadata/file_manifest.csv`
5. **历史数据完整性**: 2017-2023年历史数据已完整迁移到 `data/raw/`
