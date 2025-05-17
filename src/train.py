import torch
from model import PVT

def train_model(x_data, y_data, vocab_size, epochs=300, lr=0.01, save_path=None):
    import torch.nn as nn

    x = torch.tensor(x_data)
    y = torch.tensor(y_data).squeeze()

    model = PVT(vocab_size)
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

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    return model

def load_model(vocab_size, path):
    model = PVT(vocab_size)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model