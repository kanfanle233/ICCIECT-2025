"""
计算机视觉多任务演示平台
基于 Ultralytics YOLO + Gradio
v7.0 - 模块化重构：配置/会话/工具/推理/界面分离
"""

import os
import sys

# Windows asyncio 修复
if sys.platform == "win32":
    import asyncio
    from asyncio.proactor_events import _ProactorBasePipeTransport

    _orig = _ProactorBasePipeTransport._call_connection_lost

    def _patch(self, exc):
        """修补 Windows asyncio 的 ConnectionResetError，防止程序意外退出。"""
        try:
            _orig(self, exc)
        except ConnectionResetError:
            pass

    _ProactorBasePipeTransport._call_connection_lost = _patch

# macOS / Apple Silicon 优化
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 8))

from config import DEVICE, DEVICE_DISPLAY, DEVICE_REASON
from ui import build_interface

if __name__ == "__main__":
    print(f"[设备] 检测到计算设备: {DEVICE_DISPLAY}")
    print(f"[设备] 选择原因: {DEVICE_REASON}")
    if DEVICE == "mps":
        print("[设备] Apple Silicon MPS (Metal Performance Shaders) 加速已启用")
    elif DEVICE == "cuda":
        print("[设备] NVIDIA CUDA GPU 加速已启用")
    else:
        print("[设备] 使用 CPU 推理（这是最后的保底回退路径）")

    demo, css = build_interface()
    demo.queue(max_size=1)
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False, show_error=True, css=css)
