import torch

torch.backends.cudnn.enabled = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Active Device Status:", "cuda" if torch.cuda.is_available() else "cpu")