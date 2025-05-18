import torch
from train import train_model
from collections import Counter

# 학습용 문장들 (이거 너가 직접 바꿔도 됨)
sentences = [
    "나는 마사라야",
    "내 이름은 마사라야",
    "나는 마인크래프트를 하고 있어",
    "안녕 내 이름은 마사라야야",
    "나는 게임을 좋아해",
]

# vocab 만들기
words = []
for s in sentences:
    words.extend(s.strip().split())

vocab = list(sorted(set(words)))
special_tokens = ["<pad>", "<unk>"]
vocab = special_tokens + vocab

stoi = {word: i for i, word in enumerate(vocab)}
itos = {i: word for word, i in stoi.items()}

# 학습 데이터 만들기
seq_len = 3
x_data = []
y_data = []

for s in sentences:
    tokens = s.strip().split()
    tokens = ["<pad>"] * (seq_len - 1) + tokens
    for i in range(len(tokens) - seq_len):
        context = tokens[i:i+seq_len]
        target = tokens[i+seq_len]
        x_data.append([stoi.get(w, stoi["<unk>"]) for w in context])
        y_data.append(stoi.get(target, stoi["<unk>"]))

# 텐서로 변환
x_tensor = torch.tensor(x_data)
y_tensor = torch.tensor(y_data).unsqueeze(1)

# 학습
train_model(
    x_tensor,
    y_tensor,
    vocab_size=len(stoi),
    embed_dim=32,
    seq_len=seq_len,
    epochs=300,
    lr=0.01,
    save_path="trained_model.pvt",
    stoi=stoi,
    itos=itos
)
