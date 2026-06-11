"""
CIFAR-10 断点续训与最佳模型保存版本。

教学重点：
- epoch_logs 是“字典组成的列表”，每个字典记录一轮训练的 loss、accuracy、lr。
- MODEL_PATH_BEST 保存测试集准确率最高的 checkpoint，适合课后继续训练或评估。
- 设备自动选择 MPS -> CUDA -> CPU，CUDA 才启用混合精度 AMP。
"""

import os, time, random
import numpy as np
import pandas as pd
import torch
import torch as T
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from runtime_compat import (
    adapt_batch_size,
    autocast_context,
    build_loader_kwargs,
    build_grad_scaler,
    can_use_amp,
    cifar10_root,
    get_best_device,
    move_to_device,
    print_device_summary,
)


# ============ 0) 基础设置 ============
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============ 1) 超参数 ============
BATCH_SIZE = 8192
LR         = 1e-3
EPOCHS     = 200
WEIGHT_DECAY = 1e-4

MODEL_PATH_FINAL = "cifar10_final.pth"
MODEL_PATH_BEST  = "cifar10_best.pth"       # 按测试集精度保存最好
RESULT_CSV       = "cifar10_logs.csv"
LOSS_NPY         = "total_loss.npy"
ACC_NPY          = "total_acc.npy"
RESUME_PATH      = None                     # 想要断点续训就填已有 ckpt 路径

DATA_ROOT        = cifar10_root()


# ============ 2) 模型 ============
class Cifar10Model(T.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = T.nn.Sequential(
            # 三通道输入对应 RGB 图片；BatchNorm 帮助训练更平稳。
            T.nn.BatchNorm2d(3),
            T.nn.Conv2d(3, 32, 3, 1, 1), T.nn.BatchNorm2d(32), T.nn.ReLU(), T.nn.MaxPool2d(2, 2),
            T.nn.Conv2d(32, 64, 3, 1, 1), T.nn.BatchNorm2d(64), T.nn.ReLU(), T.nn.MaxPool2d(2, 2),
            T.nn.Conv2d(64, 128, 3, 1, 1), T.nn.BatchNorm2d(128), T.nn.ReLU(), T.nn.Dropout(0.5),
            T.nn.MaxPool2d(2, 2),
            T.nn.Flatten(),
            T.nn.Linear(128 * 4 * 4, 400),
            T.nn.BatchNorm1d(400),
            T.nn.ReLU(),
            T.nn.Linear(400, 10),
        )
    def forward(self, x):
        return self.net(x)


# ============ 3) 训练与评估 ============
@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    acc_list = []
    for images, labels in loader:
        images, labels = move_to_device(images, labels, device=device)
        logits = model(images)
        predictions = logits.argmax(dim=1)
        acc_list.append((predictions == labels).float().mean().item())
    model.train()
    return float(np.mean(acc_list)) if acc_list else 0.0


def main():
    assert __name__ == "__main__"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    set_seed(42)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    device = get_best_device()
    runtime_batch = adapt_batch_size(BATCH_SIZE, device, mps_cap=512, cpu_cap=256)
    print_device_summary(device)
    print("Batch size:", runtime_batch)

    # 2. 数据
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_tf  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    loader_kwargs = build_loader_kwargs(device, max_workers=16)
    print(f"DataLoader 参数: {loader_kwargs}")

    train_dataset = datasets.CIFAR10(root=DATA_ROOT, train=True,  download=True, transform=train_tf)
    test_dataset  = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_dataset, batch_size=runtime_batch, shuffle=True,
                              **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  batch_size=runtime_batch, shuffle=False,
                              **loader_kwargs)

    # 3. 模型/优化器/调度器/损失
    model = Cifar10Model().to(device)
    print("参数数量:", sum(p.numel() for p in model.parameters()))

    optimizer = T.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = T.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn   = T.nn.CrossEntropyLoss()
    scaler    = build_grad_scaler(device, enabled=True)
    amp_enabled = can_use_amp(device)

    start_epoch, best_acc = 0, 0.0
    # 断点续训（可选）
    if RESUME_PATH and os.path.exists(RESUME_PATH):
        ckpt = torch.load(RESUME_PATH, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["sched"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"✅ Resumed from {RESUME_PATH} @ epoch {start_epoch}, best_acc={best_acc:.4f}")

    # 4. 训练循环
    total_loss_curve, total_acc_curve, epoch_logs = [], [], []
    moving_size, loss_buf, acc_buf = 50, [], []

    print("开始训练...")
    t0 = time.time()

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        for batch_index, (images, labels) in enumerate(train_loader):
            images, labels = move_to_device(images, labels, device=device)

            with autocast_context(device, enabled=amp_enabled):
                logits = model(images)
                loss = loss_fn(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 移动窗口日志
            with torch.no_grad():
                loss_buf.append(loss.detach().float().cpu().item())
                acc_buf.append((logits.argmax(1) == labels).float().mean().item())
                if len(loss_buf) > moving_size:
                    loss_buf.pop(0); acc_buf.pop(0)
                total_loss_curve.append(float(np.mean(loss_buf)))
                total_acc_curve.append(float(np.mean(acc_buf)))

            if (batch_index + 1) % 20 == 0 or (batch_index + 1) == len(train_loader):
                print(f"Epoch {epoch+1}/{EPOCHS} | "
                      f"Step {batch_index+1}/{len(train_loader)} | "
                      f"Loss(mv): {np.mean(loss_buf):.4f} | "
                      f"Acc(mv): {np.mean(acc_buf):.4f}")

        # 每轮评估 + 调度器
        test_acc = eval_acc(model, test_loader, device)
        scheduler.step()

        # 记录 & 保存最佳
        epoch_logs.append({"epoch": epoch + 1,
                           "moving_loss": float(np.mean(loss_buf)) if loss_buf else 0.0,
                           "test_acc": float(test_acc),
                           "lr": float(optimizer.param_groups[0]["lr"])})

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"model": model.state_dict(),
                        "optim": optimizer.state_dict(),
                        "sched": scheduler.state_dict(),
                        "epoch": epoch,
                        "best_acc": best_acc}, MODEL_PATH_BEST)
            print(f"🌟 Save BEST @ epoch {epoch+1}: acc={best_acc:.4f} -> {MODEL_PATH_BEST}")

        print(f"--- Epoch {epoch+1} done | TestAcc: {test_acc:.4f} | Best: {best_acc:.4f}\n")

    # 5. 结束与保存
    torch.save(model.state_dict(), MODEL_PATH_FINAL)
    print(f"模型已保存到: {MODEL_PATH_FINAL}")

    df = pd.DataFrame(epoch_logs)
    df.to_csv(RESULT_CSV, index=False)
    print(f"每轮测试结果已保存到: {RESULT_CSV}")

    np.save(LOSS_NPY, np.array(total_loss_curve, dtype=np.float32))
    np.save(ACC_NPY,  np.array(total_acc_curve,  dtype=np.float32))
    print(f"绘图数据已保存到: {LOSS_NPY} 和 {ACC_NPY}")

    minutes = int((time.time() - t0) // 60)
    seconds = int((time.time() - t0) % 60)
    print(f"训练总耗时: {minutes} 分 {seconds} 秒")


if __name__ == "__main__":
    main()
