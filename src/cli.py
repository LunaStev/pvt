import torch
import sys
import os
import torch
import hashlib
from train import load_model
from data import build_vocab, make_dataset
from train import train_model, load_model

def hash_filename(text):
    return hashlib.md5(text.encode()).hexdigest() + ".pvt"

def is_compatible(pvt_path, embed_dim):
    try:
        state = torch.load(pvt_path)
        emb_shape = state["model_state_dict"]["embedding.weight"].shape
        return emb_shape[1] == embed_dim
    except:
        return False

def interactive_cli(model_path, embed_dim=32, seq_len=5):
    if len(sys.argv) < 2:
        print("Usage: uv run src/main.py \"your text here\"")
        return

    text = sys.argv[1]
    if len(text) < 4:
        print("Error: Text must be at least 4 characters long.")
        return

    SEQ_LEN = 5
    SPLIT_BY = "word"

    stoi, itos, encode, decode, vocab_size = build_vocab(text, split_by=SPLIT_BY)

    MODEL_PATH = hash_filename(text)

    EMBED_DIM = 32

    if os.path.exists(MODEL_PATH) and is_compatible(MODEL_PATH, vocab_size, EMBED_DIM):
        print(f"Loading model from {MODEL_PATH}")
        model, stoi, itos = load_model(MODEL_PATH, embed_dim=EMBED_DIM, seq_len=SEQ_LEN)
        vocab_size = len(itos)
        encode = lambda s: [stoi[c] for c in (s if isinstance(s, list) else s.split())]
        decode = lambda l: [itos[i] for i in l]
    else:
        print("Training new model...")
        x_data, y_data = make_dataset(text, encode, seq_len=SEQ_LEN, split_by=SPLIT_BY)
        model = train_model(
            x_data, y_data, vocab_size,
            embed_dim=EMBED_DIM,
            seq_len=SEQ_LEN,
            save_path=MODEL_PATH,
            stoi=stoi,
            itos=itos
        )

    if os.path.exists(MODEL_PATH):
        saved_vocab_size = torch.load(MODEL_PATH)["model_state_dict"]["embedding.weight"].shape[0]
        if saved_vocab_size == vocab_size:
            print(f"Loading model from {MODEL_PATH}")
            model, stoi, itos = load_model(MODEL_PATH, embed_dim=EMBED_DIM, seq_len=SEQ_LEN)
            vocab_size = len(itos)
            encode = lambda s: [stoi[c] for c in (s if isinstance(s, list) else s.split())]
            decode = lambda l: [itos[i] for i in l]
        else:
            print("Vocab size mismatch. Training new model...")
            x_data, y_data = make_dataset(text, encode, seq_len=SEQ_LEN, split_by=SPLIT_BY)
            model = train_model(
                x_data, y_data, vocab_size,
                embed_dim=EMBED_DIM,
                seq_len=SEQ_LEN,
                save_path=MODEL_PATH,
                stoi=stoi,
                itos=itos
            )
    else:
        print("Training new model...")
        x_data, y_data = make_dataset(text, encode, seq_len=SEQ_LEN, split_by=SPLIT_BY)
        model = train_model(
            x_data, y_data, vocab_size,
            embed_dim=EMBED_DIM,
            seq_len=SEQ_LEN,
            save_path=MODEL_PATH,
            stoi=stoi,
            itos=itos
        )

    tokens = text.strip().split()[-SEQ_LEN:]
    generated = list(tokens)

    for _ in range(20):
        context_encoded = torch.tensor([encode(generated[-SEQ_LEN:])])
        logits = model(context_encoded)
        pred_idx = torch.argmax(logits, dim=1).item()

        if pred_idx in itos:
            generated.append(itos[pred_idx])
        else:
            print(f"Invalid prediction index: {pred_idx}")
            break


    print(f"\nInput Text: \"{text}\"")
    print(f"Generated: \"{' '.join(generated)}\"")