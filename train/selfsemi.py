import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data.data_loader import get_dataset
from data.dataset import TabularDataset
from model.selfsemi import AE, MLP
from torch.utils.data import DataLoader
from tqdm import tqdm

from .semi import SemiTrainer


def self_to_unlabel(train_set: TabularDataset, val_set: TabularDataset):
    x_train, y_train = train_set.x.numpy(), train_set.y.numpy()
    x_val, y_val = val_set.x.numpy(), val_set.y.numpy()
    x = np.concatenate([x_train, x_val])
    y = np.concatenate([y_train, y_val])
    return TabularDataset(x, y)


class SelfSemiTrainer(SemiTrainer):
    def __init__(self, config, writer=None) -> None:
        super().__init__(config, writer)
        self.self_train_set, self.self_val_set, self.self_test_set = get_dataset(config)
        self.self_opt_alg = config['self_opt']
        self.ae = AE(self.dim, self.dim).to(self.device)
        self.self_lossfn = nn.MSELoss()
        self.self_epochs = config['self_epochs']
        self.self_train_set = self_to_unlabel(self.self_train_set, self.self_val_set)
        self.self_set_optimizer()

    def self_set_optimizer(self):
        if self.self_opt_alg == 'RMSprop':
            self.self_optimizer = optim.RMSprop(self.ae.parameters(), lr=1e-3)
        elif self.self_opt_alg == 'Adam':
            self.self_optimizer = optim.Adam(self.ae.parameters())

    def self_train(self):
        train_loader = DataLoader(self.self_train_set, self.batch_size, shuffle=True)
        for e in range(self.self_epochs):
            with tqdm(train_loader, bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}") as pbar_epoch:
                for data in pbar_epoch:
                    x, _ = data
                    x = x.to(self.device)
                    self.self_optimizer.zero_grad()
                    pred = self.ae(x)
                    loss = self.self_lossfn(x, pred)
                    loss.backward()
                    self.self_optimizer.step()
                    pbar_epoch.set_description(f"epoch[{e + 1} / {self.self_epochs}]")
                    pbar_epoch.set_postfix({'loss': loss.item()})
            self.scheduler.step()

    def train(self):
        self.self_train()
        self.model = MLP(self.dim, self.l_dim, self.ae.encoder).to(self.device)
        self.set_optimizer()
        self.semi_train()