# machine-learning_final-project

机器学习课程大作业：中文购物评论情感二分类分析。项目围绕 `shopping_comments` 数据集完成文本清洗、中文分词、特征工程、传统机器学习模型、LSTM/BERT 深度学习模型与多指标评估。

## 项目内容

- `202314109方昕哲机器学习大作业_优化版.ipynb`：本地优化版 notebook，包含硬件检测、扩展特征工程、传统模型、深度学习模型和评估可视化。
- `202314109方昕哲机器学习大作业_服务器版.ipynb`：面向 CUDA GPU 服务器运行的版本，包含服务器环境与输出保存相关配置。
- `202314109方昕哲步骤图.ipynb`：大作业流程、数据理解、实验步骤和模型对比过程记录。
- `scripts/`：可复用的辅助脚本，包括数据质量检查、传统机器学习 baseline、BiLSTM 训练、Transformer 微调、预测入口、通用数据处理函数和模型定义。
- `shopping_comments/`：中文购物评论情感分类数据集，包含训练集、验证集和测试集。

## 数据集

数据文件采用制表符分隔，主要字段包括：

- `label`：情感标签，`1` 表示正向评论，`0` 表示负向评论。
- `text_a`：评论文本。

当前数据划分：

- `train.txt`：50219 条
- `dev.txt`：6277 条
- `test.txt`：6278 条

## 主要方法

- 文本预处理：HTML 清洗、正则清洗、jieba 中文分词、停用词过滤。
- 特征工程：TF-IDF、CountVectorizer、字符级 n-gram、卡方特征选择。
- 传统模型：Logistic Regression、Linear SVM、Multinomial Naive Bayes、KNN、RandomForest、GradientBoosting、RidgeClassifier、Voting Classifier。
- 深度学习模型：LSTM、BERT 文本分类。
- 评估指标：Accuracy、Precision、Recall、F1-score、ROC-AUC、Cohen's Kappa、MCC、Specificity 等。

## 环境依赖

建议使用 Python 3.10+，安装依赖：

```bash
pip install -r requirements.txt
```

如果运行 BERT 相关章节，需要能够访问或提前缓存 Hugging Face 模型。

## 运行方式

1. 安装依赖。
2. 打开对应 notebook。
3. 按章节顺序运行。

本地 Mac 可优先运行优化版；Linux/CUDA 服务器可运行服务器版。

## 便利脚本

新增脚本位于 `scripts/`，用于把 notebook 中的关键流程拆成可复现实验入口：

```bash
# 1. 数据质量检查
python scripts/00_data_quality_check.py --data_dir shopping_comments --output_dir outputs/data_quality

# 2. 传统机器学习基线
python scripts/01_train_ml_baseline.py --data_dir shopping_comments --output_dir outputs

# 3. BiLSTM 深度学习模型；先用 --fast_dev_run 做冒烟测试
python scripts/02_train_bilstm.py --data_dir shopping_comments --output_dir outputs --epochs 1 --fast_dev_run

# 4. Transformer 微调；可把 --model_name 改成本地模型路径或 bert-base-chinese
python scripts/03_finetune_transformer.py --data_dir shopping_comments --output_dir outputs --epochs 1 --fast_dev_run

# 5. 统一预测入口
python scripts/04_predict.py --model_type ml --text "这个商品质量很好，物流也很快。"
```

主要输出保存在 `outputs/`：`metrics.json`、`classification_report.txt`、`confusion_matrix.png`、训练日志和最佳模型文件。

可选数据质量检查：

```bash
python scripts/00_data_quality_check.py
```

传统机器学习 baseline 快速验证：

```bash
python scripts/01_train_ml_baseline.py --fast_dev_run
```

BiLSTM 快速验证：

```bash
python scripts/02_train_bilstm.py --fast_dev_run
```

Transformer 快速验证需要可用的本地缓存模型或网络访问：

```bash
python scripts/03_finetune_transformer.py --fast_dev_run
```

使用已训练模型预测：

```bash
python scripts/04_predict.py --model_type ml --text "这个商品质量很好，物流也很快。"
```
