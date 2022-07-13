import numpy as np
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = torch.tensor(x).to(torch.float)
        self.y = torch.tensor(y).to(torch.float)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


class SemiDataset(Dataset):
    def __init__(self, l_x, l_y, u_x) -> None:
        super().__init__()
        self.l_x = l_x
        self.l_y = l_y
        self.u_x = u_x

    def __len__(self):
        return len(self.l_x)

    def __getitem__(self, idx):
        uidx = np.random.randint(0, len(self.u_x))
        return self.l_x[idx], self.l_y[idx], self.u_x[uidx]

