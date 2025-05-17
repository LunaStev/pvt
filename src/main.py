import sys
import os
import torch
import hashlib
from data import build_vocab, make_dataset
from train import train_model, load_model

def hash_filename(text):
    return hashlib.md5(text.encode()).hexdigest() + ".pvt"

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run src/main.py \"your text here\"")
        return

    text = sys.argv[1]
    if len(text) < 4:
        print("Error: Text must be at least 4 characters long.")
        return

    stoi, itos, encode, decode, vocab_size = build_vocab(text)

    MODEL_PATH = hash_filename(text)

    if os.path.exists(MODEL_PATH):
        saved_vocab_size = torch.load(MODEL_PATH)["embedding.weight"].shape[0]
        if saved_vocab_size == vocab_size:
            print(f"Loading model from {MODEL_PATH}")
            model = load_model(vocab_size, MODEL_PATH)
        else:
            print("Vocab size mismatch. Training new model...")
            x_data, y_data = make_dataset(text, encode)
            model = train_model(x_data, y_data, vocab_size, save_path=MODEL_PATH)
    else:
        print("Training new model...")
        x_data, y_data = make_dataset(text, encode)
        model = train_model(x_data, y_data, vocab_size, save_path=MODEL_PATH)

    context = text[-3:]
    generated = list(context)
    for _ in range(20):
        context_encoded = torch.tensor([encode(generated[-3:])])
        logits = model(context_encoded)
        predicted_index = torch.argmax(logits, dim=1).item()
        predicted_char = decode([predicted_index])
        generated.append(predicted_char)

    print(f"\nInput Text: \"{text}\"")
    print(f"Generated: \"{''.join(generated)}\"")

if __name__ == "__main__":
    main()
