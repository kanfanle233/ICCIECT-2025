# train.py
"""
CelebA 人脸属性识别 CNN 模型训练

功能说明：
1. 定义 CelebAFaceDataset 类，加载并对齐 CelebA 人脸图像
2. 定义 SimpleCNN 卷积神经网络，用于多标签二分类（Male、Eyeglasses）
3. 训练循环：前向传播 -> 计算损失 -> 反向传播 -> 更新权重
4. 支持断点续训：检测已有 best_model.pth 时在其基础上继续训练
5. 训练结束后绘制损失曲线和准确率曲线
"""

import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from celeba_utils import load_celeba_full_df, load_image, align_face_by_eyes


# --- 1. 自定义 Dataset 类 ---
class CelebAFaceDataset(Dataset):
    """CelebA 人脸属性数据集，支持图像对齐和数据增强"""

    def __init__(self, df, target_attrs, image_size=(128, 128), augment=False):
        """
        参数:
            df:            DataFrame，包含 image_id 和属性列
            target_attrs:  目标属性名列表，如 ["Male", "Eyeglasses"]
            image_size:    输出图像尺寸 (H, W)
            augment:       是否启用数据增强（随机水平翻转）
        """
        self.df = df.reset_index(drop=True)
        self.target_attrs = target_attrs
        self.image_size = image_size

        # 基础预处理：PIL Image -> Tensor [0,1]，再归一化到 [-1,1]
        base_tf = [
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std =[0.5, 0.5, 0.5]),
        ]
        if augment:
            # 训练时增加随机水平翻转，提升模型泛化能力
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),  # 50% 概率水平翻转
                *base_tf
            ])
        else:
            self.transform = T.Compose(base_tf)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """获取第 idx 个样本：对齐后的人脸图像 + 属性标签"""
        row = self.df.iloc[idx]

        img_raw = load_image(row)  # BGR 格式原始图像
        face = align_face_by_eyes(img_raw, row, output_size=self.image_size)  # 对齐裁剪并 resize

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # OpenCV BGR 转 RGB
        img_pil = Image.fromarray(face_rgb)  # numpy 数组转 PIL Image
        x = self.transform(img_pil)  # 应用预处理，输出 [3, H, W] 张量

        y_vals = row[self.target_attrs].values.astype(np.float32)  # 取目标属性值
        y = torch.from_numpy(y_vals)  # 转为 Tensor，形状 [num_classes]
        return x, y


# --- 2. 小型 CNN 模型定义 ---
class SimpleCNN(nn.Module):
    """4层卷积 + 全连接分类器，输入 3x128x128，输出 num_classes 个 logits"""

    def __init__(self, num_classes):
        super().__init__()
        # 特征提取部分：4 个卷积块，每块包含 Conv -> BatchNorm -> ReLU -> MaxPool
        self.features = nn.Sequential(
            # 3x128x128 -> 32x64x64
            nn.Conv2d(3, 32, 3, padding=1),    # 3通道输入，32通道输出，3x3卷积核，padding保持尺寸
            nn.BatchNorm2d(32),                 # 批归一化，加速收敛
            nn.ReLU(inplace=True),              # 激活函数，inplace节省内存
            nn.MaxPool2d(2),                    # 2x2 最大池化，尺寸减半

            # 32x64x64 -> 64x32x32
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 64x32x32 -> 128x16x16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 128x16x16 -> 256x8x8
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # 分类器部分：展平 -> 全连接 -> Dropout -> 输出层
        self.classifier = nn.Sequential(
            nn.Flatten(),                        # 256*8*8=16384 维向量
            nn.Linear(256 * 8 * 8, 512),         # 全连接层，16384 -> 512
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),                     # 50% Dropout 防止过拟合
            nn.Linear(512, num_classes),          # 输出层，512 -> num_classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  # 返回 logits（未经 sigmoid），配合 BCEWithLogitsLoss 使用


# --- 3. 训练主函数 ---
def main():
    # 0. 设备
    assert torch.cuda.is_available(), "没有检测到 CUDA，请检查 RTX 4060 驱动"
    device = torch.device("cuda:0")
    print("当前 GPU:", torch.cuda.get_device_name(device))

    # 1. 加载 full_df
    full_df = load_celeba_full_df()
    print("CelebA 总样本数:", len(full_df))

    # 选择属性：性别 + 眼镜
    target_attrs = ["Male", "Eyeglasses"]
    num_classes = len(target_attrs)

    # -1 / 1 -> 0 / 1（CelebA 原始标注格式转换）
    for col in target_attrs:
        full_df[col] = full_df[col].replace({-1: 0, 1: 1})

    # 按 split 划分 train/val（0=train, 1=val, 2=test）
    train_df = full_df[full_df["split"] == 0].reset_index(drop=True)
    val_df   = full_df[full_df["split"] == 1].reset_index(drop=True)
    print("Train size:", len(train_df))
    print("Val size  :", len(val_df))

    # 2. Dataset & DataLoader
    IMAGE_SIZE = (128, 128)
    BATCH_SIZE = 512
    NUM_WORKERS = 4  # Windows 下先用 0，想提速再改 4

    train_dataset = CelebAFaceDataset(train_df, target_attrs, IMAGE_SIZE, augment=True)   # 训练集启用数据增强
    val_dataset   = CelebAFaceDataset(val_df,   target_attrs, IMAGE_SIZE, augment=False)  # 验证集不增强

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)   # 训练集打乱顺序
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)  # 验证集保持顺序

    # 3. 模型
    model = SimpleCNN(num_classes).to(device)
    print(model)
    print("可训练参数量:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # 4. 损失、优化器
    criterion = nn.BCEWithLogitsLoss()  # 二元交叉熵损失（内含 sigmoid，数值更稳定）
    # 默认学习率
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # Adam 优化器，weight_decay 为 L2 正则
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)  # 每 3 个 epoch 学习率减半

    # 5. 验证函数（跟之前一样）
    def evaluate(model, loader):
        """在验证集上评估模型，返回平均损失和各属性准确率"""
        model.eval()  # 评估模式
        total_loss = 0.0
        total_samples = 0
        correct_per_attr = torch.zeros(num_classes, device=device)

        with torch.no_grad():  # 验证时不计算梯度
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)

                total_loss += loss.item() * xb.size(0)
                total_samples += xb.size(0)

                probs = torch.sigmoid(logits)       # logits -> 概率
                preds = (probs > 0.5).float()       # 概率 > 0.5 为正类
                correct_per_attr += (preds == yb).sum(dim=0)

        avg_loss = total_loss / total_samples
        acc_per_attr = (correct_per_attr / total_samples).cpu().numpy()
        return avg_loss, acc_per_attr

    # 6. 断点续训：如果存在 best_model.pth，就在它基础上继续训练
    best_acc = 0.0
    if os.path.exists("../models/best_model.pth"):
        state_dict = torch.load("../models/best_model.pth", map_location=device)
        model.load_state_dict(state_dict)
        print("✅ 检测到 best_model.pth，已加载权重，在此基础上继续训练。")

        # 可选：继续训练时把学习率稍微调低一点（比如 /2）
        for g in optimizer.param_groups:
            g["lr"] = lr * 0.5
        print(f"继续训练，学习率调整为 {lr * 0.5:g}")

        # 先在 val 上评估一次，作为当前 best_acc
        val_loss0, val_acc0 = evaluate(model, val_loader)
        best_acc = float(val_acc0.mean())
        print(f"当前加载模型在验证集上的平均准确率: {best_acc:.4f} (val_loss={val_loss0:.4f})")
    else:
        print("未找到 best_model.pth，将从随机初始化开始训练。")

    # 7. 训练循环
    EPOCHS =32
    warmup_epochs = 2
    patience = 5         # 早停耐心值（当前代码中未使用早停逻辑，仅作记录）
    best_loss = float('inf')
    no_improve = 0

    # 训练历史记录，用于绘图
    history = {
        "train_loss": [],
        "val_loss": [],
        "acc": {name: [] for name in target_attrs}
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()  # 切换到训练模式
        running_loss = 0.0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()     # 清除上一轮的梯度
            logits = model(xb)        # 前向传播
            loss = criterion(logits, yb)  # 计算损失
            loss.backward()           # 反向传播，计算梯度
            optimizer.step()          # 更新模型权重

            running_loss += loss.item() * xb.size(0)
            total += xb.size(0)

            # 显示当前 GPU 显存占用
            mem = torch.cuda.memory_allocated(device) / 1024**3
            pbar.set_postfix(loss=running_loss/total, mem=f"{mem:.2f}GB")

        scheduler.step()  # 更新学习率（StepLR 每 step_size 个 epoch 减半）

        train_loss = running_loss / total
        val_loss, val_acc_attr = evaluate(model, val_loader)

        # 记录训练历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for name, acc in zip(target_attrs, val_acc_attr):
            history["acc"][name].append(acc)

        # 保存最优模型（按验证集平均准确率）
        avg_acc = float(val_acc_attr.mean())
        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(model.state_dict(), "../models/best_model.pth")
            print(f"💾 更新并保存新的 best_model.pth, 平均 acc = {best_acc:.4f}")

        acc_str = ", ".join([f"{n}={a:.3f}" for n, a in zip(target_attrs, val_acc_attr)])
        print(f"\n[Epoch {epoch}/{EPOCHS}] "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  |  {acc_str}\n")

    print("继续训练结束。")
    print("当前最佳验证平均准确率:", best_acc)

    # --- 8. 绘制损失曲线和准确率曲线 ---
    epochs_range = range(1, len(history["train_loss"]) + 1)

    # Loss 曲线
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=200)
    plt.show()

    # 每个属性的准确率曲线
    plt.figure(figsize=(6, 4))
    for name, vals in history["acc"].items():
        plt.plot(epochs_range, vals, label=f"{name} Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy per Attribute")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("acc_curve.png", dpi=200)
    plt.show()

    print("损失曲线已保存为 loss_curve.png，准确率曲线已保存为 acc_curve.png")




# ========== Windows 多进程 DataLoader 必备入口 ==========
if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)  # Windows 下必须用 spawn 模式
    except RuntimeError:
        # 可能已经设置过 start_method，忽略即可
        pass

    main()
