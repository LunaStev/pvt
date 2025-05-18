# src/web.py
from flask import Flask, request, jsonify
import torch
from train import load_model

def start_web(model_path, embed_dim=32, seq_len=5):
    app = Flask(__name__)
    model, stoi, itos = load_model(model_path, embed_dim, seq_len)
    encode = lambda s: [stoi[c] for c in s.split()]
    decode = lambda l: [itos[i] for i in l]

    @app.route("/generate", methods=["POST"])
    def generate():
        data = request.json
        text = data.get("text", "")
        tokens = text.strip().split()
        context_encoded = torch.tensor([encode(tokens[-seq_len:])])
        logits = model(context_encoded)
        pred_idx = torch.argmax(logits, dim=1).item()
        word = itos.get(pred_idx, "<unk>")
        return jsonify({"response": word})

    print("🌐 Web server running at http://localhost:5000")
    app.run()
