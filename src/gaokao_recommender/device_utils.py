"""
设备检测工具模块
================

自动检测并返回最优计算设备（CUDA > MPS > CPU）。
"""

import torch
import platform


def get_device(verbose=True):
    """
    检测并返回最优计算设备

    优先级：CUDA > MPS (Apple Silicon) > CPU

    Parameters
    ----------
    verbose : bool
        是否打印设备信息

    Returns
    -------
    torch.device
        检测到的最优设备
    str
        设备类型字符串 ('cuda', 'mps', 'cpu')
    """

    # 检测 CUDA（NVIDIA GPU）
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
        if verbose:
            print(f"✅ 检测到 CUDA GPU")
            print(f"   设备: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA 版本: {torch.version.cuda}")
            print(f"   可用 GPU 数量: {torch.cuda.device_count()}")
            print(f"   当前设备: {torch.cuda.current_device()}")

    # 检测 MPS（Apple Silicon GPU）
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_type = "mps"
        if verbose:
            print(f"✅ 检测到 MPS (Apple Silicon GPU)")
            print(f"   系统: {platform.system()} {platform.machine()}")
            print(f"   PyTorch 版本: {torch.__version__}")

    # 回退到 CPU
    else:
        device = torch.device("cpu")
        device_type = "cpu"
        if verbose:
            print(f"⚠️  使用 CPU")
            print(f"   系统: {platform.system()} {platform.machine()}")
            print(f"   PyTorch 版本: {torch.__version__}")

    return device, device_type


def check_device_compatibility(device, verbose=True):
    """
    检查设备兼容性并打印建议

    Parameters
    ----------
    device : torch.device
        要检查的设备
    verbose : bool
        是否打印建议信息

    Returns
    -------
    bool
        设备是否可用
    """

    try:
        # 尝试在设备上创建张量
        test_tensor = torch.zeros(1).to(device)
        del test_tensor

        if verbose:
            if device.type == "cuda":
                print(f"✅ CUDA 设备兼容性检查通过")
                print(f"   建议：可使用较大的 batch_size 以充分利用 GPU")
            elif device.type == "mps":
                print(f"✅ MPS 设备兼容性检查通过")
                print(f"   建议：MPS 性能接近 CUDA，适当调整 batch_size")
            else:
                print(f"✅ CPU 设备兼容性检查通过")
                print(f"   建议：减小 batch_size 以避免内存不足")

        return True

    except Exception as e:
        if verbose:
            print(f"❌ 设备兼容性检查失败: {e}")
            print(f"   建议：回退到 CPU")
        return False


def optimize_batch_size(device, default_batch_size=64, verbose=True):
    """
    根据设备类型优化 batch_size

    Parameters
    ----------
    device : torch.device
        计算设备
    default_batch_size : int
        默认 batch_size
    verbose : bool
        是否打印优化建议

    Returns
    -------
    int
        优化后的 batch_size
    """

    if device.type == "cuda":
        # CUDA GPU：可以使用较大的 batch_size
        optimized = min(default_batch_size * 2, 256)
        if verbose:
            print(f"📊 CUDA 优化: batch_size {default_batch_size} → {optimized}")

    elif device.type == "mps":
        # MPS：性能接近 CUDA，适当增加
        optimized = min(int(default_batch_size * 1.5), 192)
        if verbose:
            print(f"📊 MPS 优化: batch_size {default_batch_size} → {optimized}")

    else:
        # CPU：保持或减小 batch_size
        optimized = max(default_batch_size // 2, 16)
        if verbose:
            print(f"📊 CPU 优化: batch_size {default_batch_size} → {optimized}")

    return optimized


def print_device_summary():
    """打印完整的设备检测摘要"""

    print("\n" + "=" * 60)
    print("设备检测摘要")
    print("=" * 60)

    device, device_type = get_device(verbose=True)
    check_device_compatibility(device, verbose=True)
    optimize_batch_size(device, verbose=True)

    print("=" * 60 + "\n")

    return device, device_type


if __name__ == "__main__":
    # 测试设备检测
    print_device_summary()
