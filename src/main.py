# src/main.py
import sys
import torch
from data import build_vocab, make_dataset
from train import train_model

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run src/main.py \"your text here\"")
        return

    text = sys.argv[1]
    if len(text) < 4:
        print("Error: Text must be at least 4 characters long.")
        return

    stoi, itos, encode, decode, vocab_size = build_vocab(text)
    x_data, y_data = make_dataset(text, encode)

    model = train_model(x_data, y_data, vocab_size)

    context = text[-3:]
    context_encoded = torch.tensor([encode(context)])
    logits = model(context_encoded)
    predicted_index = torch.argmax(logits, dim=1).item()

    print(f"\nInput Text: \"{text}\"")
    print(f"Predicted Next Character After \"{context}\": \"{decode([predicted_index])}\"")

if __name__ == "__main__":
    main()
