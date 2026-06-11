"""
CIFAR-10 服务器/长轮次训练版本。

教学重点：
- 这个版本保留较大的 batch_size 和较长 epochs，适合放到性能更强的机器上训练。
- 运行时仍会自动按 MPS -> CUDA -> CPU 选择设备，并用 adapt_batch_size 防止本地机器内存不够。
- 使用 autocast_context 和 GradScaler 时，只有 CUDA 会真正开启混合精度，MPS/CPU 保持普通精度以保证稳定。
"""

import torch
import torch as T
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import pandas as pd
import time
from runtime_compat import (
    adapt_batch_size,
    autocast_context,
    build_loader_kwargs,
    build_grad_scaler,
    cifar10_root,
    get_best_device,
    move_to_device,
    print_device_summary,
)


# ---------------- 0. 环境优化 ----------------
if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    # --- 1. 超参数设置 ---
    # 建议使用我们优化后的大batch_size以充分利用GPU
    batch_size = 8192
    lr = 0.001
    epochs = 200  # 建议至少训练20轮以上看效果
    model_path = "p091_model.pth"
    result_path = "p092_result.csv"
    loss_plot_path = 'total_loss.npy'
    acc_plot_path = 'total_acc.npy'

    # --- 2. 数据增强和转换 ---
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # --- 3. 设备选择 ---
    device = get_best_device()
    print_device_summary(device)
    batch_size = adapt_batch_size(batch_size, device, mps_cap=512, cpu_cap=256)
    print(f"当前 batch_size: {batch_size}")

    # --- 4. 数据加载 (使用num_workers优化) ---
    loader_kwargs = build_loader_kwargs(device, max_workers=8)
    print(f"DataLoader 参数: {loader_kwargs}")

    train_dataset = datasets.CIFAR10(root=cifar10_root(), train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)

    test_dataset = datasets.CIFAR10(root=cifar10_root(), train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)


    # --- 5. 建模 (定义模型结构) ---
    class Cifar10Model(T.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = T.nn.Sequential(
                # BatchNorm 稳定每层输入分布，长轮次训练时更容易收敛。
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


    # --- 6. 初始化/加载模型和优化器 ---
    model = Cifar10Model().to(device)
    optimizer = T.optim.Adam(model.parameters(), lr=lr)
    loss_fn = T.nn.CrossEntropyLoss()
    print('参数数量:', sum([p.numel() for p in model.parameters()]))

    if model_path is not None and os.path.exists(model_path):
        try:
            model.load_state_dict(T.load(model_path, map_location=device))
            print(f"成功从 {model_path} 加载模型参数")
        except Exception as e:
            print(f"从 {model_path} 加载模型失败, 错误信息: {e}")
            print("将从头开始训练。")
    else:
        print("没有发现已保存的模型，将从头开始训练。")


    # --- 7. 辅助函数 ---
    def get_acc(logits, y):
        predicted = T.argmax(logits, dim=1)
        return np.mean((predicted == y).cpu().numpy())


    def get_testing_acc():
        model.eval()
        with torch.no_grad():
            accs = []
            for images, labels in test_loader:
                images, labels = move_to_device(images, labels, device=device)
                logits = model(images)
                accs.append(get_acc(logits, labels))
        model.train()
        return np.mean(accs)


    # --- 8. 训练循环 ---
    total_loss_for_plot, total_acc_for_plot, epoch_results = [], [], []
    moving_size, losses, accs = 50, [], []

    # 使用兼容旧版PyTorch的混合精度写法
    scaler = build_grad_scaler(device, enabled=True)

    print("开始训练...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = move_to_device(images, labels, device=device)

            with autocast_context(device, enabled=True):
                logits = model(images)
                loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.cpu().item())
            accs.append(get_acc(logits, labels))
            if len(losses) > moving_size:
                losses.pop(0)
                accs.pop(0)

            moving_loss = np.mean(losses)
            moving_acc = np.mean(accs)
            total_loss_for_plot.append(moving_loss)
            total_acc_for_plot.append(moving_acc)

            if (batch_idx + 1) % 2 == 0 or batch_idx == len(train_loader) - 1:  # 减少打印频率
                print(
                    f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_loader)} | Moving Loss: {moving_loss:.4f} | Moving Acc: {moving_acc:.4f}")

        test_acc_epoch = get_testing_acc()
        epoch_results.append({'epoch': epoch + 1, 'final_loss': moving_loss, 'test_accuracy': test_acc_epoch})
        print(f"--- Epoch {epoch + 1} 结束 --- Test Accuracy: {test_acc_epoch:.4f}\n")

    end_time = time.time()
    total_training_time = end_time - start_time
    minutes = int(total_training_time // 60)
    seconds = int(total_training_time % 60)

    print('训练完毕')
    print(f"训练总耗时: {minutes} 分 {seconds} 秒")

    # --- 9. 保存所有结果 ---
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")

    df = pd.DataFrame(epoch_results)
    df.to_csv(result_path, index=False)
    print(f"每轮测试结果已保存到: {result_path}")

    np.save(loss_plot_path, total_loss_for_plot)
    np.save(acc_plot_path, total_acc_for_plot)
    print(f"绘图数据已保存到: {loss_plot_path} 和 {acc_plot_path}")
