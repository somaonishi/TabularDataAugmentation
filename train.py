import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.augment import get_transforms
from data.data_loader import get_dataset
from model import get_model
from utils import EarlyStopping, perf_metric

log = logging.getLogger(__name__)


class TabularDataset(Dataset):
    def __init__(self, set) -> None:
        super().__init__()
        self.x, self.y = set

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class Train:
    def __init__(self, config, writer=None) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.writer = writer

        self.train_set, self.val_set, self.test_set = get_dataset(config)
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.test_batch_size = config['test_batch_size']
        self.dim = self.train_set[0][0].shape[-1]
        self.l_dim = self.train_set[0][1].shape[-1]
        print(self.dim)

        self.transform = get_transforms(config['augment'], config['augmenters'])
        self.add_dim = False
        if self.transform is not None and 'add_dim' in config['augment']:
            self.dim += 1
            self.add_dim = True

        self.model = get_model(config['model_name'], self.dim, self.l_dim)
        self.model.to(self.device)

        if self.l_dim != 1:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.99999)
        self.early_stopping = EarlyStopping(patience=config['early_stopping_patience'], path='checkpoint.pt')

    def train(self):
        train_loader = DataLoader(self.train_set, self.batch_size)
        for e in range(self.epochs):
            with tqdm(total=len(train_loader), bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}") as pbar_epoch:
                self.model.train()
                all_loss = []
                for data in train_loader:
                    pbar_epoch.update(1)
                    x, y = data
                    x = x.to(self.device)
                    y = y.to(self.device)
                    # if self.transform is not None:
                    #     x = self.transform(x)

                    self.optimizer.zero_grad()
                    pred = self.model(x, self.transform)
                    loss = self.loss_fn(pred, y)
                    loss.backward()
                    self.optimizer.step()

                    all_loss.append(loss.item())
                    pbar_epoch.set_description(f"epoch[{e + 1} / {self.epochs}]")
                    pbar_epoch.set_postfix({'loss': loss.item()})
                # self.scheduler.step()
                self.model.eval()
                with torch.no_grad():
                    x, y = self.val_set.x, self.val_set.y
                    x = x.to(self.device)
                    y = y.to(self.device)
                    pred = self.model(x)
                    val_loss = self.loss_fn(pred, y)
                    pbar_epoch.set_postfix({
                        'loss': np.array(all_loss).mean(),
                        'val_loss': val_loss.item(),
                        'val_acc': perf_metric('acc', y.cpu().numpy(), pred.cpu().numpy())
                    })

                self.writer.add_scalar('val_loss', val_loss, e)
                self.early_stopping(val_loss, self.model)
                if self.early_stopping.early_stop:
                    print(f'early stopping {e} / {self.epochs}')
                    self.model.load_state_dict(torch.load('checkpoint.pt'))
                    break

    def test(self):
        test_loader = DataLoader(self.test_set, self.test_batch_size)
        results = []
        self.model.eval()
        with tqdm(test_loader, bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}") as pbar_epoch:
            for data in pbar_epoch:
                with torch.no_grad():
                    x, y = data
                    x = x.to(self.device)
                    if self.add_dim:
                        new_dim = torch.zeros(len(x), device=x.device).view(-1, 1)
                        x = torch.cat([x, new_dim], dim=1)

                    pred = self.model(x)
                    results.append(perf_metric('acc', y.cpu().numpy(), pred.cpu().numpy()))
        log.info(f'Performance: {100 * torch.tensor(results).mean()}')
        return 100 * torch.tensor(results).mean()
