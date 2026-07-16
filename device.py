import torch.cuda as cuda

device = 'cuda' if cuda.is_available() else 'cpu'
print(f"Available Device: {device}")