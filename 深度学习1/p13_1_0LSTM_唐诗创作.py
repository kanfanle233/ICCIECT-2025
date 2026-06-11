# -*- coding: utf-8 -*-
"""
唐诗 LSTM 创作训练版本。

数据结构说明：
- ch_dict: 字符到 [编号, 出现次数] 的字典，例如 {'春': [12, 5]}。
- ch_order: 按编号顺序保存字符，方便把模型预测编号转回汉字。
- samples: 每首诗对应一个整数数组，例如 [3, 10, 8, ...]。
- 训练时输入 x 是前 n-1 个字，标签 y 是后 n-1 个字，模型学习“根据前文预测下一个字”。

设备选择：自动按 MPS -> CUDA -> CPU；CUDA 才开启 AMP。
"""

import os
from pathlib import Path
import numpy as np
import torch as T
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from runtime_compat import (
    adapt_batch_size,
    autocast_context,
    build_loader_kwargs,
    build_grad_scaler,
    can_use_amp,
    get_best_device,
    move_to_device,
    print_device_summary,
)

# ============== 数据准备（内联 get_samples） ==============
DATA_DIR = Path(__file__).parent
TXT_PATH = DATA_DIR / "全唐诗_7X4.txt"

def _get_id(ch, ch_dict, ch_order):
    """给每个汉字分配唯一编号，并统计出现次数。"""
    if ch in ch_dict:
        ch_dict[ch][1] += 1
        return ch_dict[ch][0]
    idx = len(ch_dict)
    ch_dict[ch] = [idx, 1]
    ch_order.append(ch)
    return idx

def get_samples():
    """把文本文件中的诗句转成整数序列，供 Embedding 层使用。"""
    ch_dict, ch_order, samples = {}, [], []
    with open(TXT_PATH, encoding="utf-8") as fp:
        poems = [p.strip() for p in fp if p.strip()]
    for p in poems:
        ids = [_get_id(ch, ch_dict, ch_order) for ch in p]
        samples.append(np.array(ids, dtype=np.int64))
    print(f"[DATA] poems={len(poems)}, vocab={len(ch_order)}")
    return samples, ch_dict, ch_order

# ====================== 超参 ======================
EPOCHS = 5
BATCH_SIZE = 4096        # OOM 就降 256/128
EMBED_DIM = 512
HIDDEN = 1024
LAYERS = 3
DROPOUT = 0.3
LR = 2e-3
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0
MODEL_PATH = str(DATA_DIR / "p13_1_model.pth")
BEST_PATH  = str(DATA_DIR / "p13_1_best.pth")

device = get_best_device()
print_device_summary(device)
if T.cuda.is_available():
    T.backends.cudnn.benchmark = True
T.manual_seed(42)

# ====================== 模型 ======================
class PoemModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.lstm = nn.LSTM(
            input_size=EMBED_DIM, hidden_size=HIDDEN,
            num_layers=LAYERS, batch_first=True, dropout=DROPOUT
        )
        self.fc = nn.Linear(HIDDEN, vocab_size)

    def forward(self, x):
        x = self.embedding(x)      # [B, L, E]：字符编号变成可学习向量。
        x, _ = self.lstm(x)        # [B, L, H]：LSTM 记住前文上下文。
        x = self.fc(x)             # [B, L, V]：每个位置预测词表中哪个字最可能出现。
        return x

def get_model(vocab_size):
    m = PoemModel(vocab_size).to(device)
    if os.path.exists(MODEL_PATH):
        try:
            state = T.load(MODEL_PATH, map_location=device)
            m.load_state_dict(state, strict=False)
            print(f"[CKPT] Loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"[CKPT] Load failed: {e}")
    else:
        print(f"[CKPT] Not found {MODEL_PATH}, train from scratch")
    return m

# ====================== Loader ======================
def build_loader(samples):
    class PoemDataset(Dataset):
        def __len__(self): return len(samples)
        def __getitem__(self, idx):
            s = samples[idx]
            # 输入少最后一个字，标签少第一个字，二者错开一位形成“预测下一个字”任务。
            return T.tensor(s[:-1], dtype=T.long), T.tensor(s[1:], dtype=T.long)

    runtime_batch = adapt_batch_size(BATCH_SIZE, device, mps_cap=512, cpu_cap=128)
    loader_kwargs = build_loader_kwargs(device, max_workers=8)
    print(f"[DATA] runtime_batch={runtime_batch}, loader={loader_kwargs}")
    return DataLoader(PoemDataset(), batch_size=runtime_batch, shuffle=True, **loader_kwargs)

# ====================== 训练 ======================
def train_3090(samples, ch_dict, ch_order, class_weights=None):
    vocab_size = len(ch_order)
    model = get_model(vocab_size)
    dl = build_loader(samples)

    weight_t = None
    if class_weights is not None:
        weight_t = T.tensor(class_weights, dtype=T.float32, device=device)

    loss_fn = nn.CrossEntropyLoss(weight=weight_t, label_smoothing=0.05)
    optim = T.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=WEIGHT_DECAY)
    scheduler = T.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

    amp_enabled = can_use_amp(device)
    scaler = build_grad_scaler(device, enabled=amp_enabled)
    best_loss = float("inf")
    print(f"[ENV] device={device}, AMP={amp_enabled}, base_batch={BATCH_SIZE}")

    for epoch in range(EPOCHS):
        model.train()
        running, seen = 0.0, 0
        for token_ids, target_ids in dl:
            token_ids, target_ids = move_to_device(token_ids, target_ids, device=device)

            optim.zero_grad(set_to_none=True)
            with autocast_context(device, enabled=amp_enabled):
                logits = model(token_ids)                         # [B, L, V]
                loss = loss_fn(logits.reshape(-1, vocab_size), target_ids.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optim)
            scaler.update()

            running += loss.item() * token_ids.size(0)
            seen += token_ids.size(0)

        epoch_loss = running / max(seen, 1)
        scheduler.step()
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | lr={optim.param_groups[0]['lr']:.4g} | loss={epoch_loss:.6f}")

        T.save(model.state_dict(), MODEL_PATH)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            T.save(model.state_dict(), BEST_PATH)
            print(f"  ↳ Best saved: {BEST_PATH} (loss={best_loss:.6f})")
    print("训练完成。")

# ====================== 主入口 ======================
if __name__ == "__main__":
    samples, ch_dict, ch_order = get_samples()
    # 频次 -> 类别权重（可传 None 关闭）
    counts = np.array([ch_dict[ch][1] for ch in ch_order], dtype=np.float64)
    inv = 1.0 / counts
    class_weights = (inv / inv.sum()) * len(inv)   # 均值≈1
    train_3090(samples, ch_dict, ch_order, class_weights=class_weights)
