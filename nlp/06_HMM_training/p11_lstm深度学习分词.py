import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import time

# 1. 准备数据，分离样本并进行BMES标注
# 目标文件位于当前目录的"第4章"下的trainCorpus.txt
corpus_path = os.path.join(os.path.dirname(__file__), '第4章', 'trainCorpus.txt')

def load_data(filepath, max_sentences=5000):
    sentences = []
    tags = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_sentences:  # 限制样本数量以加快训练速度，可以根据需要调整
                break
            line = line.strip()
            if not line:
                continue
            words = line.split()
            char_seq = []
            tag_seq = []
            for word in words:
                if len(word) == 0:
                    continue
                if len(word) == 1:
                    char_seq.append(word)
                    tag_seq.append('S')
                else:
                    char_seq.append(word[0])
                    tag_seq.append('B')
                    for c in word[1:-1]:
                        char_seq.append(c)
                        tag_seq.append('M')
                    char_seq.append(word[-1])
                    tag_seq.append('M')
            if len(char_seq) > 0:
                sentences.append(char_seq)
                tags.append(tag_seq)
    return sentences, tags

print("正在加载并处理数据（进行BMS标注）...")
sentences, tags = load_data(corpus_path, max_sentences=5000)

# 2. 构建词典和标签字典
word2id = {'<PAD>': 0, '<UNK>': 1}
tag2id = {'<PAD>': 0, 'B': 1, 'M': 2, 'S': 3}
id2tag = {v: k for k, v in tag2id.items()}

for seq in sentences:
    for char in seq:
        if char not in word2id:
            word2id[char] = len(word2id)

def seq2id(seq, vocab):
    return [vocab.get(char, vocab.get('<UNK>', 0)) for char in seq]

# 转换为id序列
X = [seq2id(seq, word2id) for seq in sentences]
y = [seq2id(seq, tag2id) for seq in tags]

# Padding对齐处理
max_len = max(len(seq) for seq in X)
max_len = min(max_len, 100) # 防止序列过长导致内存溢出

X_pad = []
y_pad = []
for i in range(len(X)):
    if len(X[i]) > max_len:
        X_pad.append(X[i][:max_len])
        y_pad.append(y[i][:max_len])
    else:
        X_pad.append(X[i] + [0] * (max_len - len(X[i])))
        y_pad.append(y[i] + [0] * (max_len - len(y[i])))

X_tensor = torch.tensor(X_pad, dtype=torch.long)
y_tensor = torch.tensor(y_pad, dtype=torch.long)

# 3. 定义双层LSTM模型
class BiLSTMSegmenter(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=128, hidden_dim=128):
        super(BiLSTMSegmenter, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # 双层双向LSTM，每方向hidden_dim//2=64，总输出hidden_dim=128
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=2, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        return self.softmax(tag_space)

model = BiLSTMSegmenter(len(word2id), len(tag2id))
loss_function = nn.NLLLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

print("模型选择: 双层Bi-LSTM")
print(f"数据量: {len(sentences)} 条句子")
print("开始训练模型...")

epochs = 5
for epoch in range(epochs):
    start_time = time.time()
    total_loss = 0
    model.train()
    for batch_x, batch_y in dataloader:
        model.zero_grad()
        tag_scores = model(batch_x)
        # 计算损失
        loss = loss_function(tag_scores.view(-1, len(tag2id)), batch_y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}, Time: {time.time()-start_time:.2f}s")

# 5. 模型预测
def predict(sentence, model):
    model.eval()
    with torch.no_grad():
        x = torch.tensor([seq2id(list(sentence), word2id)], dtype=torch.long)
        tag_scores = model(x)
        predicted_tags = torch.argmax(tag_scores, dim=-1)[0].tolist()
        
        tags_str = [id2tag[tag] for tag in predicted_tags]
        result = []
        word = ""
        for char, tag in zip(sentence, tags_str):
            if tag == 'S':
                if word:  # 先把之前累积的词输出
                    result.append(word)
                    word = ""
                result.append(char)  # S单独成词
            elif tag == 'B':
                if word:  # 先把之前累积的词输出
                    result.append(word)
                word = char  # 开始新词
            elif tag == 'M':
                word += char  # 继续累积（包括原E，即词结束也归入M）
        if word:
            result.append(word)
        return result

# 6. 使用句子进行分词预测
test_sentence = "美军称已对伊朗南部发动新一轮空袭"
print("\n" + "="*40)
print(f"待预测的句子: {test_sentence}")
seg_result = predict(test_sentence, model)
print(f"模型预测分词结果: {' / '.join(seg_result)}")
print("="*40)
