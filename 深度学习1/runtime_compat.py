"""
跨平台运行兼容工具。

这个文件专门给 0 基础同学解决一个常见问题：
同一份 PyTorch 代码，可能在三类机器上运行。

1. Mac 芯片：优先使用 Apple 的 MPS 后端，也就是 Mac 本地神经网络加速器。
2. NVIDIA 显卡：如果没有可用 MPS，再使用 CUDA。
3. 普通电脑：如果前两者都没有，就回到 CPU。

各个训练脚本只需要调用这里的函数，不需要在每个文件里重复写设备判断。
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path

from typing import Any

import torch


def get_best_device() -> torch.device:
    """
    按“本地 Mac 加速器 -> CUDA -> CPU”的顺序选择运行设备。

    为什么先选 MPS：
    用户当前主要在 Mac 本地运行，Apple Silicon 的 MPS 是最自然的本地加速器。
    如果代码被复制到带 NVIDIA 显卡的服务器上，再自动切到 CUDA。
    最后才用 CPU，保证没有加速器时也能跑通。
    """
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def is_cuda(device: torch.device | str) -> bool:
    """判断当前设备是不是 NVIDIA CUDA。"""
    return torch.device(device).type == "cuda"


def is_mps(device: torch.device | str) -> bool:
    """判断当前设备是不是 Apple Silicon 的 MPS。"""
    return torch.device(device).type == "mps"


def can_use_amp(device: torch.device | str) -> bool:
    """
    是否开启自动混合精度 AMP。

    AMP 可以让 CUDA 显卡用更快的低精度计算。
    MPS/CPU 的 AMP 兼容性更容易随 PyTorch 版本变化，所以教学脚本里默认关闭，
    先保证“稳定跑通”，再追求极限速度。
    """
    return is_cuda(device)


def describe_device(device: torch.device | str) -> str:
    """给学生看的设备说明，用于 print 输出。"""
    device = torch.device(device)
    if device.type == "mps":
        return "mps (Apple Silicon/Mac 本地神经网络加速器)"
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NVIDIA GPU"
        return f"cuda ({name})"
    return "cpu (没有检测到可用加速器，使用普通处理器)"


def print_device_summary(device: torch.device | str | None = None) -> None:
    """打印一次清楚的设备选择结果，适合训练脚本启动时调用。"""
    selected = get_best_device() if device is None else torch.device(device)
    print(f"当前使用的设备: {describe_device(selected)}")


def adapt_batch_size(
    base_batch_size: int,
    device: torch.device | str,
    *,
    mps_cap: int = 512,
    cpu_cap: int = 256,
) -> int:
    """
    根据设备自动压低过大的 batch_size。

    batch_size 表示“一次喂给模型多少张图片/多少条样本”。
    数值越大，训练可能越快，但越容易显存或内存不够。
    CUDA 服务器通常显存更大，所以保留原值；MPS/CPU 做安全封顶。
    """
    base_batch_size = max(int(base_batch_size), 1)
    dev = torch.device(device).type
    if dev == "cuda":
        return base_batch_size
    if dev == "mps":
        return min(base_batch_size, mps_cap)
    return min(base_batch_size, cpu_cap)


def build_loader_kwargs(
    device: torch.device | str,
    *,
    max_workers: int = 4,
    workers: int | None = None,
    persistent_workers: bool = True,
) -> dict:
    """
    返回 DataLoader 的推荐参数。

    DataLoader 负责“从数据集中按批次取样本”。
    CUDA 服务器上可以开多个 worker 加速读数据；Mac/MPS/CPU 上多进程更容易遇到
    spawn、共享内存或权限问题，所以默认 worker=0，优先保证课堂代码稳定。
    """
    if workers is None:
        if os.name == "nt":
            workers = 0
        elif is_mps(device) or not is_cuda(device):
            workers = 0
        else:
            workers = min(os.cpu_count() or 1, max_workers)

    workers = max(int(workers), 0)
    kwargs = {
        "num_workers": workers,
        "pin_memory": is_cuda(device),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
    return kwargs


def autocast_context(device: torch.device | str, enabled: bool = True):
    """根据设备返回 autocast 上下文；非 CUDA 时返回空上下文。"""
    if not enabled or not can_use_amp(device):
        return nullcontext()

    try:
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    except Exception:
        return torch.cuda.amp.autocast()


def build_grad_scaler(device: torch.device | str, enabled: bool = True):
    """根据设备返回 GradScaler；非 CUDA 返回 disabled scaler。"""
    use_amp = enabled and can_use_amp(device)

    try:
        return torch.cuda.amp.GradScaler(enabled=use_amp)
    except Exception:
        # 极老版本兜底：构造一个最小可用对象
        class _NoOpScaler:
            def scale(self, loss):
                return loss

            def unscale_(self, *_args, **_kwargs):
                return None

            def step(self, optimizer):
                optimizer.step()

            def update(self):
                return None

        return _NoOpScaler()


def move_to_device(*values: Any, device: torch.device | str, non_blocking: bool | None = None):
    """
    把一批张量统一搬到运行设备。

    例子：
        images, labels = move_to_device(images, labels, device=device)

    non_blocking 只在 CUDA pinned memory 时真正有意义；MPS/CPU 下保持 False 更稳。
    """
    selected = torch.device(device)
    if non_blocking is None:
        non_blocking = is_cuda(selected)

    moved_values = []
    for value in values:
        if hasattr(value, "to"):
            moved_values.append(value.to(selected, non_blocking=non_blocking))
        else:
            moved_values.append(value)

    if len(moved_values) == 1:
        return moved_values[0]
    return tuple(moved_values)


def project_dir() -> Path:
    """当前项目根目录，也就是这些脚本所在的文件夹。"""
    return Path(__file__).resolve().parent


def mnist_root() -> str:
    """MNIST 数据集目录；可用环境变量 MNIST_ROOT 临时覆盖。"""
    return os.environ.get("MNIST_ROOT", str(project_dir() / "data"))


def cifar10_root() -> str:
    """CIFAR-10 数据集目录；可用环境变量 CIFAR10_ROOT 临时覆盖。"""
    return os.environ.get("CIFAR10_ROOT", str(project_dir() / "cifar-10"))
