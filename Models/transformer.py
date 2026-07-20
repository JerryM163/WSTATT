import torch

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1_1 = torch.nn.Conv2d(in_channels, out_channels)

class Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

class TRANS_STATT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.encoder = Encoder()

        self.decoder = Decoder()

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)

        out = self.decoder(x, x1, x2, x3)

        return out

class TRANS_WSTATT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.encoder = Encoder()

        self.decoder = Decoder()

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)

        out = self.decoder(x, x1, x2, x3)

        return out