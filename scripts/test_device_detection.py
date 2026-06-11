#!/usr/bin/env python3
"""
测试设备检测功能
================

验证 MPS、CUDA、CPU 的自动检测和优化。
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gaokao_recommender.device_utils import (
    get_device,
    check_device_compatibility,
    optimize_batch_size,
    print_device_summary
)

def test_device_detection():
    """测试设备检测功能"""

    print("=" * 60)
    print("测试 1: 基本设备检测")
    print("=" * 60)

    device, device_type = get_device(verbose=True)

    assert device_type in ['cuda', 'mps', 'cpu'], f"未知设备类型: {device_type}"
    print(f"✅ 设备检测通过: {device_type}")

    print("\n" + "=" * 60)
    print("测试 2: 设备兼容性检查")
    print("=" * 60)

    is_compatible = check_device_compatibility(device, verbose=True)
    assert is_compatible, "设备兼容性检查失败"
    print(f"✅ 设备兼容性检查通过")

    print("\n" + "=" * 60)
    print("测试 3: Batch Size 优化")
    print("=" * 60)

    for default_bs in [16, 32, 64]:
        optimized_bs = optimize_batch_size(device, default_batch_size=default_bs, verbose=True)
        assert optimized_bs > 0, f"优化后的 batch_size 应该大于 0"
        print(f"✅ Batch size 优化通过: {default_bs} → {optimized_bs}")

    print("\n" + "=" * 60)
    print("测试 4: 完整设备摘要")
    print("=" * 60)

    device, device_type = print_device_summary()
    assert device is not None, "设备不应为 None"
    assert device_type in ['cuda', 'mps', 'cpu'], f"未知设备类型: {device_type}"
    print(f"✅ 完整设备摘要测试通过")

    return device, device_type


def test_pytorch_operations(device, device_type):
    """测试 PyTorch 在检测到的设备上的操作"""

    print("\n" + "=" * 60)
    print("测试 5: PyTorch 张量操作")
    print("=" * 60)

    import torch

    # 创建测试张量
    x = torch.randn(100, 10).to(device)
    y = torch.randn(100, 10).to(device)

    # 基本操作
    z = x + y
    z = torch.mm(x.T, y)

    # 验证设备（处理 mps:0 vs mps 的情况）
    z_device_type = z.device.type
    expected_device_type = device.type if hasattr(device, 'type') else device
    assert z_device_type == expected_device_type, f"张量设备类型不匹配: {z_device_type} != {expected_device_type}"
    print(f"✅ 张量创建和基本操作通过")

    # 测试模型
    print("\n" + "=" * 60)
    print("测试 6: 简单神经网络")
    print("=" * 60)

    model = torch.nn.Sequential(
        torch.nn.Linear(10, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    ).to(device)

    # 前向传播
    input_tensor = torch.randn(32, 10).to(device)
    output = model(input_tensor)

    output_device_type = output.device.type
    assert output_device_type == expected_device_type, f"输出设备类型不匹配: {output_device_type} != {expected_device_type}"
    print(f"✅ 神经网络前向传播通过")

    # 反向传播
    loss = output.mean()
    loss.backward()

    print(f"✅ 反向传播通过")

    # 测试数据加载器优化
    print("\n" + "=" * 60)
    print("测试 7: DataLoader 配置")
    print("=" * 60)

    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(input_tensor, torch.randn(32, 1).to(device))
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        pin_memory=(device_type == "cuda")
    )

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=(device_type == "cuda"))
        batch_y = batch_y.to(device, non_blocking=(device_type == "cuda"))
        print(f"✅ DataLoader 迭代通过")
        break

    print(f"\n✅ 所有 PyTorch 操作测试通过！")


def main():
    """主测试函数"""

    print("🧪 开始设备检测测试...\n")

    try:
        # 测试设备检测
        device, device_type = test_device_detection()

        # 测试 PyTorch 操作
        test_pytorch_operations(device, device_type)

        print("\n" + "=" * 60)
        print("🎉 所有测试通过！")
        print("=" * 60)
        print(f"\n设备信息:")
        print(f"  类型: {device_type}")
        print(f"  设备: {device}")
        print(f"\n建议:")
        if device_type == "cuda":
            print("  ✅ 使用 CUDA GPU 加速")
            print("  ✅ 可以增大 batch_size 以提高性能")
        elif device_type == "mps":
            print("  ✅ 使用 Apple Silicon (MPS) 加速")
            print("  ✅ 性能接近 CUDA，适当调整 batch_size")
        else:
            print("  ⚠️  使用 CPU")
            print("  ⚠️  建议减小 batch_size 以避免内存问题")
        print()

        return 0

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
