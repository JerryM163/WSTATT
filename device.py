import torch.cuda as cuda

device = 'CUDA' if cuda.is_available() else 'CPU'
print(f"Available Device: {device}")