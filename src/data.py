def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return stoi, itos, encode, decode, len(chars)

def make_dataset(text, encode):
    x_data = []
    y_data = []
    for i in range(len(text) - 3):
        x = encode(text[i:i+3])
        y = encode(text[i+3])
        x_data.append(x)
        y_data.append(y)
    return x_data, y_data