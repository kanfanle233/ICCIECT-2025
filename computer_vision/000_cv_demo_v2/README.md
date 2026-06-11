# 计算机视觉多任务演示平台

基于 **Python + Gradio + Ultralytics YOLO** 的离线演示系统。

## 功能特性

- **五种核心视觉任务**：目标检测、图像分割、图像分类、姿态估计、定向检测
- **三种输入方式**：本地上传文件、实时摄像头、预置开源样本
- **离线运行**：模型首次运行时自动下载，之后无需联网
- **中文界面**：所有标签、结果均支持中文显示
- **Apple Silicon 加速**：macOS 上自动启用 MPS (Metal Performance Shaders) 推理加速

## 环境要求

- Python >= 3.8
- CUDA（可选，NVIDIA GPU 加速推理）
- Apple Silicon (M1/M2/M3/M4) 自动启用 MPS 加速

## 快速开始

```bash
# 1. 创建虚拟环境
conda create -n cv_demo_env python=3.10

# 2. 激活虚拟环境
conda activate cv_demo_env

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行演示
python main.py

# 5. 在浏览器中访问
# 网址：http://localhost:7861
```

首次运行时会自动下载 YOLO 预训练模型，请确保网络畅通。
模型下载后缓存在本地，后续运行无需联网。

### macOS / Apple Silicon 加速

程序会自动检测 Apple Silicon (M1/M2/M3/M4) 芯片并启用 MPS 加速，推理速度比 CPU 快 2-5 倍。

手动控制设备：
```bash
# 强制使用 CPU
CV_DEVICE=cpu python main.py

# 强制使用 MPS（默认自动检测）
CV_DEVICE=mps python main.py

# 如果你在带 NVIDIA 显卡的机器上教学
CV_DEVICE=cuda python main.py
```

#### 可选：Core ML / ANE 加速（更快的推理速度）

如需利用 Apple Neural Engine 获得 3-8 倍加速，可以将模型导出为 Core ML 格式：

```bash
pip install coremltools
python -c "from utils import export_to_coreml; export_to_coreml('detection')"
```

导出后会在当前目录生成 `.mlpackage` 文件，使用 `YOLO("模型.mlpackage")` 直接加载推理。

## 项目结构

```
000_cv_demo_v2/
├── main.py             # 主程序入口
├── config.py           # 配置与设备检测
├── inference.py        # 推理引擎
├── session.py          # 会话管理
├── ui.py               # Gradio 界面
├── utils.py            # 工具函数
├── requirements.txt    # 依赖列表
├── README.md           # 说明文档
├── yolo11n*.pt         # YOLO11 预训练模型
└── samples/
    ├── images/         # 预置图片样本
    └── videos/         # 预置视频样本
```

## 任务说明

| 任务     | 模型文件        | 输出                   |
| -------- | --------------- | ---------------------- |
| 目标检测 | yolo11n.pt      | 边界框 + 类别 + 置信度 |
| 图像分割 | yolo11n-seg.pt  | 像素级掩膜 + 边界框    |
| 图像分类 | yolo11n-cls.pt  | Top-5 类别概率         |
| 姿态估计 | yolo11n-pose.pt | 17 个人体关键点        |
| 定向检测 | yolo11n-obb.pt  | 带角度边界框           |

## 常见问题

**Q: 首次运行很慢？**
A: 首次运行需要下载模型权重文件，请耐心等待。下载完成后自动缓存。

**Q: 如何添加自定义样本？**
A: 将图片放入 `samples/images/` 目录，视频放入 `samples/videos/` 目录，重启程序即可在预置样本列表中看到。

**Q: 摄像头无法调用？**
A: 请确保浏览器已授予摄像头权限，且系统摄像头未被其他程序占用。macOS 用户请检查 系统设置 > 隐私与安全性 > 摄像头 中已授权对应浏览器。

**Q: macOS 上如何确认 MPS 加速已启用？**
A: 启动程序后查看终端输出，应显示 `[设备] 检测到计算设备: mps` 和 `Apple Silicon MPS 加速已启用`。

---

本演示仅供计算机视觉教学使用。
