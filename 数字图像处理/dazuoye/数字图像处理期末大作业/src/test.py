# test.py
"""
CelebA 人脸属性识别模型测试与可视化

功能说明：
1. 加载已训练的 SimpleCNN 模型，在 CelebA 测试集（split==2）上评估
2. 计算每个属性（Male、Eyeglasses）的准确率和平均损失
3. 可视化若干张测试图片的预测结果与真值对比
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from celeba_utils import load_celeba_full_df, load_image, align_face_by_eyes
from train import CelebAFaceDataset, SimpleCNN   # 直接复用 Dataset 和 模型结构


# --- 1. 评估函数 ---
def evaluate(model, loader, device, criterion):
    """在给定数据集上评估模型，返回平均损失和每个属性的准确率"""
    model.eval()  # 切换到评估模式，关闭 Dropout 和 BatchNorm 的训练行为
    total_loss = 0.0
    total_samples = 0
    num_classes = model.classifier[-1].out_features  # 从模型最后一层获取类别数
    correct_per_attr = torch.zeros(num_classes, device=device)  # 每个属性的正确计数

    with torch.no_grad():  # 禁用梯度计算，节省显存
        for xb, yb in tqdm(loader, desc="Testing"):
            xb, yb = xb.to(device), yb.to(device)  # 数据搬到 GPU

            logits = model(xb)  # 前向传播，输出原始 logits
            loss = criterion(logits, yb)  # 计算二元交叉熵损失

            total_loss += loss.item() * xb.size(0)  # 累加总损失
            total_samples += xb.size(0)  # 累加样本数

            probs = torch.sigmoid(logits)  # sigmoid 将 logits 转为概率 [0, 1]
            preds = (probs > 0.5).float()  # 概率 > 0.5 判为正类
            correct_per_attr += (preds == yb).sum(dim=0)  # 逐属性统计正确数

    avg_loss = total_loss / total_samples  # 平均损失
    acc_per_attr = (correct_per_attr / total_samples).cpu().numpy()  # 各属性准确率
    return avg_loss, acc_per_attr


# --- 2. 预测可视化函数 ---
def visualize_predictions(model, device, test_df, target_attrs, num_samples=12):
    """
    从 test_df 抽 num_samples 张图，可视化 GT & 预测

    每张图显示真实属性标签（GT）和模型预测概率（P）
    """
    model.eval()

    # 图像预处理：ToTensor 转为 [0,1] 张量，Normalize 归一化到 [-1,1]
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    samples = test_df.sample(num_samples, random_state=42)  # 随机抽样，固定种子可复现

    rows = 3
    cols = int(np.ceil(num_samples / rows))  # 计算网格列数

    plt.figure(figsize=(4 * cols, 4 * rows))

    with torch.no_grad():
        for i, (_, row) in enumerate(samples.iterrows()):
            img_raw = load_image(row)  # BGR 格式原始图像
            face = align_face_by_eyes(img_raw, row, output_size=(128, 128))  # 对齐并裁剪人脸
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # BGR 转 RGB 用于 matplotlib 显示

            # 预处理并推理
            x = tf(Image.fromarray(face_rgb)).unsqueeze(0).to(device)  # 增加 batch 维度
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0]  # 各属性的预测概率

            # 真值（0/1）
            gt = row[target_attrs].values.astype(int)

            plt.subplot(rows, cols, i + 1)
            plt.imshow(face_rgb)
            plt.axis("off")
            title = (
                f"GT: M={gt[0]}, G={gt[1]}\n"   # GT = Ground Truth（真值）
                f"P : M={probs[0]:.2f}, G={probs[1]:.2f}"  # P = Prediction（预测概率）
            )
            plt.title(title, fontsize=10)

    plt.tight_layout()
    plt.savefig("test_visualization.png", dpi=200)
    plt.show()
    print("可视化结果已保存到 test_visualization.png")


# --- 3. 主测试流程 ---
def main():
    # 设备选择
    assert torch.cuda.is_available(), "没有检测到 CUDA，请检查 GPU 环境"
    device = torch.device("cuda:0")
    print("当前 GPU:", torch.cuda.get_device_name(device))

    # 1. 读取 full_df，并准备 test_df（split == 2）
    full_df = load_celeba_full_df()
    print("CelebA 总样本数:", len(full_df))

    # 目标属性：性别（Male）和是否戴眼镜（Eyeglasses）
    target_attrs = ["Male", "Eyeglasses"]

    # CelebA 原始标注为 -1/1，转换为 0/1 便于二分类
    for col in target_attrs:
        full_df[col] = full_df[col].replace({-1: 0, 1: 1})

    # split==2 对应测试集
    test_df = full_df[full_df["split"] == 2].reset_index(drop=True)
    print("Test size:", len(test_df))

    # 2. 构建 Dataset / DataLoader
    IMAGE_SIZE = (128, 128)  # 人脸图像统一尺寸
    BATCH_SIZE = 512
    NUM_WORKERS = 0  # Windows 稳定起见用 0

    test_dataset = CelebAFaceDataset(test_df, target_attrs, IMAGE_SIZE, augment=False)  # 测试集不做数据增强
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)  # pin_memory 加速 CPU->GPU 传输

    # 3. 构建模型并加载最优权重
    num_classes = len(target_attrs)
    model = SimpleCNN(num_classes).to(device)

    state_dict = torch.load("../models/best_model.pth", map_location=device)  # 加载训练好的权重
    model.load_state_dict(state_dict)
    print("已加载 best_model.pth")

    criterion = torch.nn.BCEWithLogitsLoss()  # 二元交叉熵损失（带 sigmoid）

    # 4. 在 test 集上评估
    test_loss, test_acc_attr = evaluate(model, test_loader, device, criterion)

    print("\n=== Test Result ===")
    print(f"Test Loss: {test_loss:.4f}")
    for name, acc in zip(target_attrs, test_acc_attr):
        print(f"{name} accuracy: {acc:.4f}")
    print(f"Mean accuracy: {float(test_acc_attr.mean()):.4f}")

    # 5. 抽样可视化若干张图片
    visualize_predictions(model, device, test_df, target_attrs, num_samples=12)


# --- 4. Windows 多进程入口 ---
if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)  # Windows 下 DataLoader 多进程需要 spawn 模式
    except RuntimeError:
        pass

    main()
