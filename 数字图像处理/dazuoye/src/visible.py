# visible.py
"""
CelebA 人脸属性识别模型全面评估与可视化

功能说明：
1. 加载训练好的模型，在测试集上进行推理
2. 输出每个属性的分类报告（Precision、Recall、F1-score）
3. 计算并保存指标汇总表（CSV 格式）
4. 绘制混淆矩阵热力图
5. 可视化正确预测和错误预测的样本

涉及知识点：
- PyTorch 模型推理（torch.no_grad）
- sklearn 分类评估指标
- matplotlib/seaborn 可视化
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文字体（Windows）
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题



import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from sklearn.metrics import confusion_matrix, classification_report

# Import your project code
from celeba_utils import load_celeba_full_df, load_image, align_face_by_eyes
from train import CelebAFaceDataset, SimpleCNN


# --- 0. 设备选择 ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --- 1. 加载数据集 ---
full_df = load_celeba_full_df()
target_attrs = ["Male", "Eyeglasses"]  # 目标属性：性别、是否戴眼镜

# CelebA 原始标注 -1/1 转换为 0/1
for col in target_attrs:
    full_df[col] = full_df[col].replace({-1: 0, 1: 1})

# split==2 对应测试集
test_df = full_df[full_df["split"] == 2].reset_index(drop=True)
print("Test size:", len(test_df))

IMAGE_SIZE = (128, 128)
test_dataset = CelebAFaceDataset(test_df, target_attrs, IMAGE_SIZE, augment=False)  # 测试集不做数据增强
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


# --- 2. 加载训练好的模型 ---
model = SimpleCNN(num_classes=2).to(device)
state_dict = torch.load("../models/best_model.pth", map_location=device)  # 加载最优权重
model.load_state_dict(state_dict)
model.eval()  # 切换到评估模式
print("Loaded best_model.pth")


# --- 3. 在测试集上进行推理 ---
all_labels = []  # 存储所有真实标签
all_preds = []   # 存储所有预测结果
all_probs = []   # 存储所有预测概率

with torch.no_grad():  # 推理时禁用梯度计算
    for xb, yb in test_loader:
        xb = xb.to(device)

        logits = model(xb)                       # 前向传播得到 logits
        probs = torch.sigmoid(logits).cpu().numpy()  # sigmoid 转概率并搬到 CPU
        preds = (probs > 0.5).astype(int)        # 概率 > 0.5 判为正类
        yb = yb.cpu().numpy()

        all_labels.append(yb)
        all_preds.append(preds)
        all_probs.append(probs)

# 按 batch 维度拼接
all_labels = np.vstack(all_labels)  # 形状: (N, num_classes)
all_preds = np.vstack(all_preds)
all_probs = np.vstack(all_probs)

print("Inference complete.")


# --- 4. 分类报告（Precision / Recall / F1） ---
print("\n================ Classification Report ================")
for i, attr in enumerate(target_attrs):
    print(f"\n=== {attr} ===")
    print(classification_report(all_labels[:, i], all_preds[:, i], digits=4))

# --- 4.5 指标汇总表（Accuracy / Precision / Recall / F1） ---
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

rows = []
for i, attr in enumerate(target_attrs):
    y_true = all_labels[:, i]
    y_pred = all_preds[:, i]

    acc = accuracy_score(y_true, y_pred)  # 整体准确率

    # 二分类：取正类（label=1）的 Precision、Recall、F1
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average="binary",
        pos_label=1,
        zero_division=0  # 避免除零警告
    )

    rows.append({
        "Attribute": attr,
        "Accuracy": acc,
        "Precision": p,
        "Recall": r,
        "F1-score": f1
    })

metrics_df = pd.DataFrame(rows)

print("\n================ Metrics Summary (per attribute) ================")
print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# 保存为 CSV，方便直接贴到论文表格里
metrics_df.to_csv("metrics_summary.csv", index=False, float_format="%.6f")
print("Metrics summary saved as metrics_summary.csv")

# --- 5. 混淆矩阵可视化（每个属性单独绘制） ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, attr in enumerate(target_attrs):
    cm = confusion_matrix(all_labels[:, i], all_preds[:, i])  # 计算混淆矩阵
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",       # 热力图可视化
                xticklabels=["0","1"], yticklabels=["0","1"], ax=axes[i])
    axes[i].set_title(f"{attr} Confusion Matrix")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("True")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.show()
print("Confusion matrix saved as confusion_matrix.png")


# --- 6. 可视化正确预测的样本 ---
correct_idx = np.where((all_labels == all_preds).all(axis=1))[0]  # 所有属性都预测正确的样本索引
sample_correct = random.sample(correct_idx.tolist(), min(12, len(correct_idx)))  # 最多抽 12 张

tf = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)  # 归一化到 [-1, 1]
])

plt.figure(figsize=(12, 6))
for i, idx in enumerate(sample_correct):
    row = test_df.iloc[idx]
    img_raw = load_image(row)  # BGR 格式
    face = align_face_by_eyes(img_raw, row, output_size=(128,128))  # 对齐裁剪
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # 转 RGB 用于 matplotlib 显示

    gt = all_labels[idx]    # 真实标签
    pred = all_preds[idx]   # 预测标签

    plt.subplot(3, 4, i+1)
    plt.imshow(rgb)
    plt.axis("off")
    plt.title(f"✓ Correct\nGT: {gt}\nPred: {pred}", fontsize=9)

plt.tight_layout()
plt.savefig("correct_samples.png", dpi=200)
plt.show()
print("Correct samples saved as correct_samples.png")


# --- 7. 可视化错误预测的样本 ---
wrong_idx = np.where((all_labels != all_preds).any(axis=1))[0]  # 至少一个属性预测错误的样本

if len(wrong_idx) == 0:
    print("🎉 测试集中没有错误预测！模型极其优秀！")
else:
    sample_wrong = random.sample(wrong_idx.tolist(), min(12, len(wrong_idx)))  # 最多抽 12 张

    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(sample_wrong):
        row = test_df.iloc[idx]
        img_raw = load_image(row)
        face = align_face_by_eyes(img_raw, row, output_size=(128,128))
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        gt = all_labels[idx]
        pred = all_preds[idx]

        plt.subplot(3, 4, i+1)
        plt.imshow(rgb)
        plt.axis("off")
        plt.title(f"❌ Wrong\nGT: {gt}\nPred: {pred}", fontsize=9)

    plt.tight_layout()
    plt.savefig("wrong_samples.png", dpi=200)
    plt.show()
    print("Wrong samples saved as wrong_samples.png")
