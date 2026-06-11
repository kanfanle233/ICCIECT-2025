"""
唐诗 LSTM 基础训练版本。

核心数据结构：
- ch_dict: 每个汉字对应的编号和出现次数。
- samples: 每首诗转换后的编号序列。
- PoemDataset 返回 (前文, 下一个字标签)，让模型学习语言续写。
"""

import torch.nn as nn
import torch as T
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from p13_0_LSTM_唐诗创作_样本处理 import get_samples
from runtime_compat import build_loader_kwargs, get_best_device, move_to_device, print_device_summary

epochs = 10
batch_size = 20
lr = 0.01
losses_size = 20
model_path = 'p13_1_model.pth'
device = get_best_device()
print_device_summary(device)

input_size = 20
num_layers = 2
dropout = 0.3
samples, ch_dict, ch_order = get_samples()
weights = [ch_dict[ch][1] for ch in ch_order]
weights = 1 / np.array(weights)
weights = weights / np.sum(weights) * len(weights)

# 1. 样本
class PoemDataset(Dataset):
    def __len__(self):
        return len(samples)

    def __getitem__(self, item):
        s = samples[item]
        # x 是前 n-1 个字，y 是后 n-1 个字；二者错一位，用来训练“预测下一个字”。
        return s[:-1], s[1:]

# 2. 模型
class PoemModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(ch_order), input_size)
        self.lstm = nn.LSTM(input_size=input_size,
                    hidden_size=input_size,
                    num_layers=num_layers,
                    bidirectional=False,
                    dropout=dropout,
                    batch_first=True)
        self.line = nn.Linear(input_size, len(ch_dict))

    def forward(self, x):
        x = self.embedding(x)          # [B, L] -> [B, L, input_size]
        x, (long, short) = self.lstm(x)
        x = self.line(x)               # [B, L, vocab_size]
        return x

# 3. 训练
def get_model():
    model = PoemModel().to(device)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(T.load(model_path, map_location=device))
            print(f'从{model_path}加载模型成功')
        except Exception as exc:
            print(f'从{model_path}加载模型失败，模型很可能改变了：{exc}')
    else:
        print(f'未发现模型{model_path}')
    return model

def train():
    model = get_model()
    optim = T.optim.Adam(model.parameters(), lr=lr)
    losses = []
    loss_fn = nn.CrossEntropyLoss(T.from_numpy(weights).float().to(device))
    loader_kwargs = build_loader_kwargs(device, max_workers=4)
    dl = DataLoader(PoemDataset(), batch_size, shuffle=True, **loader_kwargs)
    print(f"DataLoader 参数: {loader_kwargs}")
    for epoch in range(epochs):
        trained = 0
        for batch, (token_ids, target_ids) in enumerate(dl):
            model.train()
            token_ids, target_ids = move_to_device(token_ids, target_ids, device=device)
            logits = model(token_ids)
            logits = T.reshape(logits, [-1, len(ch_dict)])
            target_ids = T.reshape(target_ids, [-1])
            loss = loss_fn(logits, target_ids)

            loss.backward()
            optim.step()
            optim.zero_grad()

            model.eval()
            with T.no_grad():
                trained += len(token_ids)
                losses.append(loss.item())
                if (batch+1) % losses_size == 0:
                    loss = np.mean(losses)
                    print(f'{epoch+1}_{trained/len(samples)*100:3.0f}% loss = {loss:.8f}')
                    losses.clear()
        T.save(model.state_dict(), model_path)
        print(f'模型保存至{model_path}')
    print('训练结束')

# 4. 测试
def ids2poem(ids):
    poem = [ch_order[id] for id in ids]
    return ''.join(poem)

# 5. 生成函数（插入版）
def _test(ch='天'):
    poem = ch
    ch = ch_dict[ch][0]
    model = get_model()
    model.eval()
    with T.no_grad():
        x = np.array([[ch]], dtype=np.int64)   # [1, 1]
        x = move_to_device(T.from_numpy(x), device=device)
        C = move_to_device(T.from_numpy(np.zeros([num_layers, 1, input_size], dtype=np.float32)), device=device)
        h = move_to_device(T.from_numpy(np.zeros([num_layers, 1, input_size], dtype=np.float32)), device=device)

        for i in range(32 - 1):
            x = model.embedding(x)  # [1, 1, 100]
            y, (C, h) = model.lstm(x, (C, h))
            y = model.line(y)       # [1, 1, 4340]
            y = T.argmax(y, 2)      # [1, 1]
            x = y                   # x, y: [batch_size, seq_len]
            y = y.cpu().numpy()
            poem += ch_order[y[0, 0]]
    print(poem)

if __name__ == '__main__':
    train()
    # 调用示例：
    # _test('天')
