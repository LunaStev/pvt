import torch
from torch import nn
from model import PVT

def train_model(x_data, y_data, vocab_size, epochs=300, lr=0.01):
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

    return model