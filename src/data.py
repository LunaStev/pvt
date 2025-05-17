import torch

def build_vocab(text: str, split_by="word"):
    if split_by == "word":
        tokens = text.strip().split()
    else:
        tokens = list(text.strip())

    vocab = sorted(set(tokens))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in enumerate(vocab)}
    encode = lambda s: [stoi[c] for c in (s if isinstance(s, list) else s.split())]
    decode = lambda l: ' '.join([itos[i] for i in l])

    return stoi, itos, encode, decode, len(vocab)


def make_dataset(text: str, encode, seq_len=5, split_by="word"):
    if split_by == "word":
        tokens = text.strip().split()
    else:
        tokens = list(text.strip())

    x_data = []
    y_data = []
    for i in range(len(tokens) - seq_len):
        x = encode(tokens[i:i + seq_len])
        y = encode(tokens[i + seq_len])
        x_data.append(x)
        y_data.append(y)

    return torch.tensor(x_data), torch.tensor(y_data).squeeze()