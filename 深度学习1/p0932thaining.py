"""
CIFAR-10 已训练模型评估脚本。

本脚本不重新训练，只读取 checkpoint，在测试集上计算准确率和混淆矩阵。
数据结构：
- all_preds: 保存所有 batch 的预测类别。
- all_labels: 保存所有 batch 的真实类别。
最后用 concatenate 拼成完整数组，方便计算整体准确率。
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from runtime_compat import (
    adapt_batch_size,
    build_loader_kwargs,
    cifar10_root,
    get_best_device,
    move_to_device,
    print_device_summary,
)

try:
    from sklearn.metrics import classification_report, confusion_matrix
except Exception:
    classification_report = None
    confusion_matrix = None

# ========= 0) 路径与设备 =========
# 把这里换成你下载的 pth 文件名
CKPT_PATH = r"cifar10_best.pth"
DATA_ROOT = cifar10_root()   # 优先使用本地 CIFAR-10；缺失时 torchvision 才会下载。
DEVICE = get_best_device()

# ========= 1) 模型结构（与训练时一致）=========
class Cifar10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 模型结构必须和训练脚本一致，否则 checkpoint 参数无法正确加载。
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, 10),
        )
    def forward(self, x): return self.net(x)

# ========= 2) 测试集（与训练时同样的归一化）=========
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

BATCH = 512  # CPU 上 256~1024 都可以，根据你机器性能调整

def main():
    print_device_summary(DEVICE)
    runtime_batch = adapt_batch_size(BATCH, DEVICE, mps_cap=512, cpu_cap=256)
    loader_kwargs = build_loader_kwargs(DEVICE, max_workers=4)
    print("dataloader:", loader_kwargs)

    test_dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=test_tf)
    test_loader = DataLoader(test_dataset, batch_size=runtime_batch, shuffle=False,
                             **loader_kwargs)

    # ===== 3) 加载模型 =====
    model = Cifar10Model().to(DEVICE)
    state = torch.load(CKPT_PATH, map_location=DEVICE)

    # 兼容两种保存方式：直接 state_dict 或 {'model':..., 'epoch':..., 'best_acc':...}
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)

    model.eval()

    # ===== 4) 评估 =====
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for images, labels in test_loader:
            images = move_to_device(images, device=DEVICE)
            logits = model(images)
            predictions = logits.argmax(1).cpu().numpy()
            all_preds.append(predictions)
            all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    acc = (y_true == y_pred).mean()
    print(f"\n✅ Test Top-1 Accuracy: {acc*100:.2f}%")

    classes = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
    if classification_report is not None:
        print("\nPer-class metrics:")
        print(classification_report(y_true, y_pred, target_names=classes, digits=4))
    else:
        print("\n未安装 scikit-learn，跳过 classification_report。")

    # ===== 5) 混淆矩阵图 =====
    if confusion_matrix is not None:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7,6))
        plt.imshow(cm, interpolation='nearest')
        plt.title('Confusion Matrix (CIFAR-10)')
        plt.colorbar()
        plt.xticks(range(10), classes, rotation=45)
        plt.yticks(range(10), classes)
        plt.tight_layout()
        plt.savefig("cm.png", dpi=150)
        print("混淆矩阵已保存：cm.png")

if __name__ == "__main__":
    # Windows 下避免多进程问题
    import multiprocessing as mp
    mp.freeze_support()
    main()
