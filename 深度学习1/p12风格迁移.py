"""
Neural Style Transfer 自动图片版。

教学重点：
- 内容图 content 表示“保留什么结构”，这里自动取 MNIST 的一个数字。
- 风格图 style 表示“模仿什么纹理/颜色”，这里优先使用本地 style_auto.jpg。
- Gram 矩阵用于描述风格特征之间的相关性，是风格迁移的核心数据结构。
- 设备自动选择 MPS -> CUDA -> CPU。
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from runtime_compat import get_best_device, mnist_root, move_to_device, print_device_summary

try:
    import requests
except Exception:
    requests = None

device = get_best_device()
print_device_summary(device)


# ========== 自动准备图片 ==========
def prepare_images():
    """自动准备内容图（MNIST 数字）和风格图（下载或使用本地文件）。"""
    # 1. 内容图：MNIST 数字
    mnist = datasets.MNIST(root=mnist_root(), train=False, download=True, transform=transforms.ToTensor())
    img, label = mnist[0]
    img = img.squeeze(0).mul(255).byte().numpy()
    content_img = Image.fromarray(img).convert("RGB")
    content_path = "content_auto.jpg"
    content_img.save(content_path)
    print(f"✅ 内容图已生成：{content_path} (MNIST 数字 {label})")

    # 2. 风格图：下载梵高《星空》
    style_url = "https://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg"
    style_path = "style_auto.jpg"
    if not os.path.exists(style_path):
        if requests is None:
            raise RuntimeError("缺少 requests 且本地没有 style_auto.jpg，无法自动下载风格图。")
        try:
            r = requests.get(style_url, timeout=10)
            r.raise_for_status()
            with open(style_path, "wb") as f:
                f.write(r.content)
            print(f"✅ 风格图已下载：{style_path}")
        except Exception as e:
            raise RuntimeError(f"风格图下载失败，请手动放置 style_auto.jpg。错误: {e}") from e
    else:
        print(f"✅ 已检测到现有风格图：{style_path}")
    return content_path, style_path

# ========== 图像预处理 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def image_loader(path, size=512):
    """加载图片并转成标准化的 Tensor，增加 batch 维度。"""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    image = Image.open(path).convert('RGB')
    # unsqueeze(0) 增加 batch 维度：[3, H, W] -> [1, 3, H, W]
    return move_to_device(transform(image).unsqueeze(0), device=device)

def to_pil(t):
    """将标准化后的 Tensor 还原为 PIL Image，方便显示。"""
    postpa = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/s for s in IMAGENET_STD]),
        transforms.Normalize(mean=[-m for m in IMAGENET_MEAN], std=[1., 1., 1.]),
    ])
    img = t.clone().cpu().squeeze(0)
    img = postpa(img)
    img = torch.clamp(img, 0, 1)
    return transforms.ToPILImage()(img)

# ========== Gram矩阵 & VGG提取 ==========
def gram_matrix(feat):
    """计算 Gram 矩阵，用来描述一张图的风格纹理。"""
    B, C, H, W = feat.size()
    f = feat.view(B, C, -1)
    return torch.bmm(f, f.transpose(1, 2)) / (C * H * W)

class VGGFeatures(nn.Module):
    """从 VGG19 预训练模型中提取指定层的特征，用于计算内容损失和风格损失。"""

    def __init__(self, content_layers, style_layers):
        super().__init__()
        try:
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
            print("✅ 使用预训练 VGG19 权重")
        except Exception as e:
            print(f"⚠️ 预训练权重加载失败，退回随机初始化 VGG19: {e}")
            vgg = models.vgg19(weights=None).features.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.map = self._map_layers()

    def _map_layers(self):
        mapping, block, conv_in_block = {}, 1, 0
        for i, m in enumerate(self.vgg):
            if isinstance(m, nn.Conv2d):
                conv_in_block += 1
                mapping[i] = f"conv{block}_{conv_in_block}"
            elif isinstance(m, nn.MaxPool2d):
                block += 1
                conv_in_block = 0
            elif isinstance(m, nn.ReLU):
                self.vgg[i] = nn.ReLU(inplace=False)
        return mapping

    def forward(self, x):
        c_feats, s_feats = {}, {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            name = self.map.get(i, None)
            if name:
                if name in self.content_layers:
                    c_feats[name] = x
                if name in self.style_layers:
                    s_feats[name] = x
        return c_feats, s_feats

# ========== 主流程 ==========
def run_style_transfer():
    content_path, style_path = prepare_images()

    content = image_loader(content_path, size=256)
    style = image_loader(style_path, size=256)
    # input_img 是真正被优化的图片，一开始复制内容图，之后逐步加上风格。
    input_img = content.clone().requires_grad_(True)

    content_layers = ["conv4_2"]
    style_layers = ["conv1_1","conv2_1","conv3_1","conv4_1","conv5_1"]
    vgg_feat = VGGFeatures(content_layers, style_layers).to(device)

    # 提取目标特征
    with torch.no_grad():
        t_c, _ = vgg_feat(content)
        _, t_s = vgg_feat(style)
        t_s_grams = {k: gram_matrix(v) for k, v in t_s.items()}

    optimizer = optim.Adam([input_img], lr=0.02)
    num_steps = 300
    content_w, style_w, tv_w = 1, 1e4, 1e-5

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        c, s = vgg_feat(input_img)

        c_loss = sum(nn.functional.mse_loss(c[k], t_c[k]) for k in content_layers)
        s_loss = sum(nn.functional.mse_loss(gram_matrix(s[k]), t_s_grams[k]) for k in style_layers)
        tv_loss = torch.mean(torch.abs(input_img[:, :, 1:, :] - input_img[:, :, :-1, :])) + \
                  torch.mean(torch.abs(input_img[:, :, :, 1:] - input_img[:, :, :, :-1]))

        loss = content_w * c_loss + style_w * s_loss + tv_w * tv_loss
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"[{step:3d}/300] content={c_loss.item():.4f}, style={s_loss.item():.4f}, total={loss.item():.4f}")

    out_img = to_pil(input_img.detach())
    plt.imshow(out_img)
    plt.title("MNIST 风格迁移结果")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    run_style_transfer()
