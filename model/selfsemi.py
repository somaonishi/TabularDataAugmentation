import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_dim, h_dim)

    def forward(self, x: torch.Tensor, transform=None):
        return x
        x = F.relu(self.l1(x))
        if transform is not None:
            x = transform(x)
        return x


class Decoder(nn.Module):
    def __init__(self, h_dim, out_dim) -> None:
        super().__init__()
        self.l1 = nn.Linear(h_dim, out_dim)

    def forward(self, z):
        return self.l1(z)


class AE(nn.Module):
    def __init__(self, in_dim, h_dim) -> None:
        super().__init__()
        self.encoder = Encoder(in_dim, h_dim)
        self.decoder = Decoder(h_dim, in_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_tilde = self.decoder(z)
        return x_tilde


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, encoder: nn.Module, h_dim=128) -> None:
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.l1 = nn.Linear(in_dim, h_dim)
        self.l2 = nn.Linear(h_dim, h_dim)
        self.l3 = nn.Linear(h_dim, out_dim)

    def forward(self, x: torch.Tensor, transform=None):
        x = self.encoder(x, transform)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)
