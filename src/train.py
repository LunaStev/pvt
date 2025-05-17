import torch
from model import PVT

def train_model(x_data, y_data, vocab_size, embed_dim=32, seq_len=3, epochs=300, lr=0.01, save_path=None, stoi=None, itos=None):
    from model import PVT
    import torch.nn as nn

    x = x_data.clone().detach()
    y = y_data.clone().detach().squeeze()

    model = PVT(vocab_size, embedding_dim=embed_dim, seq_len=seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

    if save_path and stoi and itos:
        torch.save({
            'model_state_dict': model.state_dict(),
            'stoi': stoi,
            'itos': itos,
        }, save_path)

    return model

def load_model(path, embed_dim=32, seq_len=5):
    checkpoint = torch.load(path)
    itos = checkpoint['itos']
    vocab_size = len(itos)
    model = PVT(vocab_size, embedding_dim=embed_dim, seq_len=seq_len)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['stoi'], checkpoint['itos']
