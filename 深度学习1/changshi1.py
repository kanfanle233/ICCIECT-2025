# -*- coding: utf-8 -*-
"""
唐诗 LSTM 评估与生成脚本。

这个文件主要用于加载已经训练好的唐诗模型，然后生成诗句。
重要数据结构：
- ch2id: 汉字 -> 编号，用来把用户输入的前缀转成模型输入。
- id2ch/ch_order: 编号 -> 汉字，用来把模型预测结果转回文字。
- freq_mask: 低频字屏蔽表，减少模型生成生僻或不稳定字符。
"""

import os
import numpy as np
import torch as T
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from p13_0_LSTM_唐诗创作_样本处理 import get_samples
from runtime_compat import build_loader_kwargs, get_best_device, move_to_device, print_device_summary

# ================= 设备 =================
device = get_best_device()
print_device_summary(device)
T.manual_seed(42)
np.random.seed(42)

# ================= 超参数 =================
batch_size   = 64
model_path   = 'p13_1_model_best.pth'

# ✅ 与已训练模型保持一致
embed_dim    = 256
hidden_dim   = 512
num_layers   = 2
dropout      = 0.2

# 仅用于评估时的 class weight（不会训练）
def _build_weights(ch_dict, ch_order, pow_smooth=0.5):
    # 用 1/sqrt(freq) 平滑（比 1/freq 温和，减少罕见字过度放大）
    freq = np.array([ch_dict[c][1] for c in ch_order], dtype=np.float32)
    w = 1.0 / np.maximum(freq, 1.0)**pow_smooth
    w = w / w.sum() * len(w)
    return w

# ================= 数据 =================
samples, ch_dict, ch_order = get_samples()
print(f"📘 共 {len(samples)} 首诗，字表大小 {len(ch_order)}")
weights = _build_weights(ch_dict, ch_order, pow_smooth=0.5)

class PoemDataset(Dataset):
    def __len__(self): return len(samples)
    def __getitem__(self, idx):
        s = samples[idx]
        # 输入和标签错开一位，形成“用前文预测下一个字”的监督学习任务。
        return T.tensor(s[:-1], dtype=T.long), T.tensor(s[1:], dtype=T.long)

# ================= 模型定义 =================
class PoemModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)  # [B, L] -> [B, L, embed_dim]
        x, _ = self.lstm(x)    # [B, L, hidden_dim]
        return self.fc(x)      # [B, L, vocab_size]

# ================= 加载模型（严格优先，失败再非严格） =================
def get_model():
    vocab_size = len(ch_order)
    model = PoemModel(vocab_size).to(device)
    if not os.path.exists(model_path):
        print(f'⚠️ 未发现模型 {model_path}')
        return model
    state = T.load(model_path, map_location=device)
    try:
        model.load_state_dict(state, strict=True)
        print(f'✅ 从 {model_path} 严格加载成功')
    except Exception as e:
        print(f'⚠️ 严格加载失败：{e}\n→ 尝试 strict=False')
        model.load_state_dict(state, strict=False)
        print('✅ 非严格加载成功（部分参数跳过）')
    return model

# ================= loss 检测（输出“无权重/加权”两套口径） =================
@T.inference_mode()
def evaluate_loss(model):
    """
    A. 无权重（token-level CE）
    B. 加权（1/sqrt(freq)）
    """
    model.eval()
    loader_kwargs = build_loader_kwargs(device, max_workers=4)
    dl = DataLoader(PoemDataset(), batch_size=batch_size, shuffle=False, **loader_kwargs)
    print(f"DataLoader 参数: {loader_kwargs}")

    # A. 无权重
    ce_plain = nn.CrossEntropyLoss(reduction='sum').to(device)
    loss_sum_plain, tok_plain = 0.0, 0

    # B. 加权
    w = T.from_numpy(weights).float().to(device)
    ce_weighted = nn.CrossEntropyLoss(weight=w, reduction='sum').to(device)
    loss_sum_weighted, tok_weighted = 0.0, 0

    for token_ids, target_ids in dl:
        token_ids, target_ids = move_to_device(token_ids, target_ids, device=device)
        logits = model(token_ids)  # [B, L, V]

        loss_plain = ce_plain(logits.transpose(1, 2), target_ids)
        loss_sum_plain += loss_plain.item()
        tok_plain += target_ids.numel()

        loss_weight = ce_weighted(logits.transpose(1, 2), target_ids)
        loss_sum_weighted += loss_weight.item()
        tok_weighted += target_ids.numel()

    avg_plain = loss_sum_plain / max(1, tok_plain)
    ppl_plain = float(np.exp(min(20.0, avg_plain)))

    avg_weighted = loss_sum_weighted / max(1, tok_weighted)
    ppl_weighted = float(np.exp(min(20.0, avg_weighted)))

    print(f"📊 无权重  loss={avg_plain:.4f}, PPL={ppl_plain:.2f}")
    print(f"📊 加权后  loss={avg_weighted:.4f}, PPL={ppl_weighted:.2f}")
    return (avg_plain, ppl_plain), (avg_weighted, ppl_weighted)

# ================= 生成（仅优化输出，不改模型） =================
@T.inference_mode()
def generate(model, ch2id, id2ch,
             prefix="春风",
             max_len=40,
             # —— 解码参数（均衡预设）——
             temperature=0.80,         # 较活，但有退火
             top_k=40,
             top_p=0.92,
             repetition_penalty=1.30,
             no_repeat_ngram_size=5,
             lines=4, line_len=7,
             # —— 频次屏蔽 ——
             min_freq=8,
             # —— 温度退火 ——
             use_temp_anneal=True):
    """
    输出优化策略：
    1) 温度退火（0.85→0.60）+ Top-k/p；
    2) 动态重复惩罚：全局已出现字、最近窗口（K=5）更强；
    3) 低频字屏蔽（min_freq）；
    4) 5-gram 禁复；禁止“立刻重复上一个字”；
    5) 7字/行，并在分行时修补连续相同尾字。
    """
    model.eval()
    ids = [ch2id.get(c, 0) for c in prefix]

    # 频次掩码
    freq = T.tensor([ch_dict[c][1] for c in ch_order], dtype=T.float32, device=device)
    freq_mask = (freq >= float(min_freq)).float().unsqueeze(0)  # [1, V]

    # 可扩展的禁止集合（目前全 1）
    ban_mask = T.ones((1, len(ch_order)), dtype=T.float32, device=device)

    def apply_topk_topp(probs):
        # probs: [1, V]
        if top_k and top_k > 0:
            v, _ = T.topk(probs, k=min(int(top_k), probs.size(-1)))
            cutoff = v[..., -1, None]
            probs = T.where(probs < cutoff, T.zeros_like(probs), probs)
        if top_p and top_p < 1.0:
            sp, si = T.sort(probs, dim=-1, descending=True)
            csum = T.cumsum(sp, dim=-1)
            mask = csum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0]  = False
            sp = sp.masked_fill(mask, 0.0)
            probs = T.zeros_like(probs).scatter(1, si, sp)
        return probs

    def violates_ngram(seq, nxt, n):
        if n <= 1 or len(seq) < n - 1:
            return False
        ngram = seq[-(n-1):] + [nxt]
        for i in range(len(seq) - (n-1)):
            if seq[i:i+n] == ngram:
                return True
        return False

    recent_k = 5  # 最近窗口长度
    for t in range(len(ids), max_len):
        x = T.tensor([ids], dtype=T.long, device=device)
        logits = model(x)[:, -1, :]                       # [1, V]
        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        logits = T.clamp(logits, min=-1e3, max=1e3)

        # 温度退火：0.85 -> 0.60
        temp = temperature
        if use_temp_anneal and max_len > 1:
            hi, lo = 0.85, 0.60
            alpha = min(1.0, t / float(max_len-1))
            temp = lo + (hi - lo) * (1.0 - alpha)
        logits = logits / max(1e-6, float(temp))

        probs = T.softmax(logits, dim=-1)
        probs = probs * freq_mask * ban_mask

        # —— 细化去重：禁止立刻重复上一个字 ——
        if len(ids) > 0:
            probs[0, ids[-1]] = 0.0

        # 动态重复惩罚（全局 + 最近窗口）
        if repetition_penalty and repetition_penalty > 1.0:
            seen = T.tensor(ids, dtype=T.long, device=device).unique()
            probs[:, seen] /= repetition_penalty
            recent = ids[-recent_k:] if len(ids) >= recent_k else ids
            if recent:
                probs[:, T.tensor(recent, device=device)] /= (repetition_penalty * 1.25)

        probs = apply_topk_topp(probs)
        probs = T.clamp(probs, min=1e-12)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # 结合 no-repeat-ngrams 约束，必要时重采
        for _ in range(12):
            nxt = int(T.multinomial(probs, num_samples=1).item())
            if not violates_ngram(ids, nxt, no_repeat_ngram_size):
                break
            probs[0, nxt] = 0.0
            s = probs.sum(dim=-1, keepdim=True)
            if s.item() <= 1e-12:
                probs = T.softmax(logits, dim=-1)
                probs = T.clamp(probs, min=1e-12)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                break
        ids.append(nxt)

    # —— 7 字/行组句，并避免行尾连续同字（稳一点的修补）——
    text = ''.join(id2ch[i] for i in ids)
    lines_out, p = [], 0
    last_tail = None
    while p < len(text) and len(lines_out) < lines:
        seg = text[p:p+line_len]
        if not seg: break
        if last_tail and len(seg) > 0 and seg[-1] == last_tail:
            # 优先从该行中找一个不罕见且不同的字替代尾字（从靠后往前找）
            replaced = False
            for j in range(line_len-2, -1, -1):
                cj = seg[j]
                if cj != last_tail and ch_dict.get(cj, (None, 0))[1] >= max(3, min_freq):
                    seg = seg[:line_len-1] + cj
                    replaced = True
                    break
            if not replaced:
                # 保底：从该行反向找第一个“与 last_tail 不同且不罕见”的字当尾字
                for cj in seg[::-1]:
                    if cj != last_tail and ch_dict.get(cj, (None, 0))[1] >= max(3, min_freq):
                        seg = seg[:line_len-1] + cj
                        break
        last_tail = seg[-1]
        lines_out.append(seg)
        p += line_len

    if not lines_out:
        return text[:lines*line_len]
    if len(lines_out) == 1:
        return lines_out[0]
    return '，\n'.join(lines_out[:-1]) + '。\n' + lines_out[-1]

# ================= 主入口 =================
if __name__ == '__main__':
    model = get_model()
    ch2id = {c: i for i, c in enumerate(ch_order)}
    id2ch = ch_order

    # 评估（打印两套口径，方便对齐历史）
    evaluate_loss(model)

    # 仅优化“输出”
    poem = generate(
        model, ch2id, id2ch,
        prefix="春风",
        max_len=40,
        temperature=0.80,
        top_k=40,
        top_p=0.92,
        repetition_penalty=1.30,
        no_repeat_ngram_size=5,
        lines=4, line_len=7,
        min_freq=8,
        use_temp_anneal=True
    )

    print("\n📝 生成样例：")
    print(poem)
