"""
环境检查脚本。

给 0 基础同学的用法：
运行 `python 检查.py`，先确认当前 Python、PyTorch 和加速器状态。
本项目训练脚本的设备选择顺序是：
MPS(Mac 本地神经网络加速器) -> CUDA(NVIDIA 显卡) -> CPU。
"""

import os
import platform

from runtime_compat import describe_device, get_best_device

# --- 1. 检查 PyTorch 是否安装 ---
try:
    import torch
except Exception as e:
    print("未检测到 PyTorch：", e)
    raise SystemExit(1)

# --- 2. 检查 Python 和 PyTorch 版本 ---
print("Python               =", platform.python_version())
print("torch.__version__    =", torch.__version__)
print("torch.version.cuda   =", torch.version.cuda)

# --- 3. 检查 Apple MPS 后端 ---
mps_backend = getattr(torch.backends, "mps", None)
mps_built = bool(mps_backend and mps_backend.is_built())
mps_available = bool(mps_backend and mps_backend.is_available())
print("MPS built?           =", mps_built)
print("MPS available?       =", mps_available)

# --- 4. 检查 NVIDIA CUDA ---
cuda_available = torch.cuda.is_available()
print("CUDA available?      =", cuda_available)
print("CUDA GPU count       =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA GPU name        =", torch.cuda.get_device_name(0))

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))

# --- 5. 输出推荐设备 ---
device = get_best_device()
print("Recommended device   =", describe_device(device))
print("\n说明：如果 MPS/CUDA 都不可用，脚本会自动回到 CPU，功能不变，只是速度更慢。")
