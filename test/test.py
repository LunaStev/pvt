import torch

checkpoint = torch.load("ab98af2c2714f983ce4fc9e8c183b734.pvt")
print("STOI keys:", list(checkpoint['stoi'].keys()))