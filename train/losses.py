import torch
import torch.nn as nn


class NormalLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.mean(torch.var(x, 0))


class FixMatchLoss(nn.Module):
    def __init__(self, t=0.95) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.t = t

    def forward(self, weak, strong):
        weak = torch.softmax(weak, dim=1)
        weak_label = torch.argmax(weak, dim=1)
        loss = self.criterion(strong, weak_label)
        loss *= weak.max(dim=1)[0] > self.t
        return loss.mean()
