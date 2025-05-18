import torch
import sys
import os
import torch
import hashlib
from train import load_model
from data import build_vocab, make_dataset
from train import train_model, load_model

def interactive_cli(model_path, embed_dim=32, seq_len=3):
    model, stoi, itos = load_model(model_path, embed_dim, seq_len)
    encode = lambda tokens: [stoi.get(c, stoi.get("<unk>", 0)) for c in tokens]
    decode = lambda l: [itos[i] for i in l]

    print("🧠 Start interactive PVT (end when 'exit' is entered)\n")
    context = input("🧍 You: ")
    while context.strip().lower() != "exit":
        tokens = context.strip().split()
        if len(tokens) < seq_len:
            padding = [list(stoi.keys())[0]] * (seq_len - len(tokens))
            tokens = padding + tokens
        context_encoded = torch.tensor([encode(tokens[-seq_len:])])
        logits = model(context_encoded)
        pred_idx = torch.argmax(logits, dim=1).item()

        if pred_idx in itos:
            next_word = itos[pred_idx]
        else:
            next_word = "<unk>"

        print(f"🤖 PVT: {next_word}")
        context = input("🧍 You: ")
