"""
LSTM 唐诗创作模型，基于字符级语言模型自动生成古诗。
教学重点：
1) Embedding 词嵌入层：将字符 ID 映射为稠密向量
2) LSTM 长短期记忆网络：捕捉序列中的长期依赖关系
3) 自回归生成：每次输入上一个字符的预测结果，逐步生成整首诗
4) 加权交叉熵损失：根据字符频率调整损失权重，平衡常见字和生僻字
"""
import torch.nn as nn
import torch as T
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from p13_0_LSTM_唐诗创作_样本处理 import get_samples

# --- 1. 设备检测与超参数 ---
# ── 设备检测（优先 MPS，其次 CUDA，最后 CPU）──────────────
if T.backends.mps.is_available():
    device = T.device("mps")
    print("使用 MPS (Apple Silicon GPU) 加速")
elif T.cuda.is_available():
    device = T.device("cuda")
    print("使用 CUDA 加速")
else:
    device = T.device("cpu")
    print("使用 CPU 训练")

epochs = 2
batch_size = 10
lr = 0.01
losses_size = 20      # 每隔多少个 batch 打印一次损失
model_path = 'p13_1_model.pth'

input_size = 100       # Embedding 和 LSTM 的向量维度
num_layers = 2         # LSTM 层数
dropout = 0            # LSTM 层间 Dropout

# --- 2. 准备样本与词频权重 ---
samples, ch_dict, ch_order = get_samples()
# 计算字符频率权重：出现越多的字，权重越低（1 - 频率占比）
weights = [ch_dict[ch][1] for ch in ch_order]
weights = np.array(weights)
weights = T.from_numpy(1 - weights / np.sum(weights)).to(device)  # 高频字权重小，低频字权重大

# --- 3. 自定义数据集 ---
# 1. 样本
class PoemDataset(Dataset):
    """唐诗数据集：每首诗的前 n-1 个字作为输入，后 n-1 个字作为目标"""
    def __len__(self):
        return len(samples)

    def __getitem__(self, item):
        s = samples[item]
        return s[:-1], s[1:]  # 输入序列 vs 目标序列（错位一位）

# --- 4. LSTM 语言模型 ---
# 2. 模型
class PoemModel(nn.Module):
    """字符级 LSTM 语言模型：Embedding -> LSTM -> Linear"""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(ch_order), input_size)  # 词嵌入
        self.lstm = nn.LSTM(input_size=input_size,
                    hidden_size=input_size,
                    num_layers=num_layers,
                    bidirectional=False,       # 单向 LSTM
                    dropout=dropout)
        self.line = nn.Linear(input_size, len(ch_dict))  # 输出层：预测下一个字符

    def forward(self, x):   # x: [batch_size, 31]
        x = self.embedding(x)   # [batch_size, 31, 100] 字符 ID -> 嵌入向量
        x, (long, short) = self.lstm(x) # x:[?, 31, 100], long: 长期记忆, short: 短期记忆
        x = self.line(x)        # [batch_size, 31, vocab_size] 预测每个位置的下一个字符
        return x

# --- 5. 模型加载 ---
# 3. 训练
def get_model():
    """加载或创建模型"""
    model = PoemModel().to(device)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(T.load(model_path, weights_only=True, map_location=device))
            print(f'从{model_path}加载模型成功')
        except:
            print(f'从{model_path}加载模型失败，模型很可能改变了')
    else:
        print(f'未发现模型{model_path}')
    return model

# --- 6. 训练函数 ---
def train():
    """训练 LSTM 语言模型：预测序列中每个位置的下一个字符"""
    model = get_model()
    optim = T.optim.Adam(model.parameters(), lr=lr)
    losses = []
    loss_fn = nn.CrossEntropyLoss(weights).to(device)  # 加权交叉熵损失
    dl = DataLoader(PoemDataset(), batch_size, True)    # 按 batch 加载数据
    for epoch in range(epochs):
        trained = 0
        for batch, (x, y) in enumerate(dl):
            model.train()
            x, y = x.to(device), y.to(device)
            p = model(x)                                  # 前向传播
            y = nn.functional.one_hot(y, len(ch_dict)).float()  # 目标转为独热
            p = T.reshape(p, [-1, len(ch_dict)])          # 展平为 [batch*seq, vocab]
            y = T.reshape(y, [-1, len(ch_dict)])
            loss = loss_fn(p, y)                           # 计算损失

            loss.backward()
            optim.step()
            optim.zero_grad()

            model.eval()
            with T.no_grad():
                trained += len(x)
                losses.append(loss.item())
                if (batch+1) % losses_size == 0:
                    loss = np.mean(losses)
                    print(f'{epoch+1}_{trained/len(samples)*100:3.0f}% loss = {loss:.8f}')
                    losses.clear()
        T.save(model.state_dict(), model_path)
        print(f'模型保存至{model_path}')
    print('训练结束')

# --- 7. 测试：自回归生成古诗 ---
# 4. 测试

def ids2poem(ids):
    """将整数 ID 序列转回诗句字符串"""
    poem = [ch_order[id] for id in ids]
    return ''.join(poem)

def _test(ch = '天'):
    """自回归生成古诗：给定起始字，逐字生成后续内容"""
    poem = ch
    ch = ch_dict[ch][0]  # 起始字的 ID
    model = get_model()
    model.eval()
    with T.no_grad():
        x = np.array([[ch]], dtype=np.int64)   # [1, 1] 初始输入
        x = T.from_numpy(x).to(device)
        # 初始化 LSTM 的隐状态（长期记忆 C 和短期记忆 h）
        C = T.from_numpy(np.zeros([num_layers, 1, input_size], dtype=np.float32)).to(device)
        h = T.from_numpy(np.zeros([num_layers, 1, input_size], dtype=np.float32)).to(device)

        for i in range(32 - 1):  # 逐字生成31个字符
            x = model.embedding(x)  # [1, 1] --> [1, 1, 100] 字符嵌入
            y, (C, h) = model.lstm(x, (C, h))  # LSTM 前向传播，更新隐状态
            y = model.line(y)   # [1, 1, 100] --> [1, 1, vocab_size] 预测下一字
            y = T.argmax(y, 2)  # [1, 1, vocab_size] --> [1, 1] 取概率最大的字
            x = y               # 将预测结果作为下一步输入（自回归）
            y = y.cpu().numpy()
            poem += ch_order[y[0, 0]]  # 拼接到诗句中
    print(poem)

if __name__ == '__main__':
    # pds = PoemDataset()
    # print(pds[123])
    # print(pds[123].shape)
    # print(ids2poem(pds[123]))

    # train()
    _test()
