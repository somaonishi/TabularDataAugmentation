import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import perf_metric

from train.base import BaseTrainer

log = logging.getLogger(__name__)


class SupervisedTrainer(BaseTrainer):
    def __init__(self, config, writer=None) -> None:
        super().__init__(config, writer)

    def train(self):
        train_loader = DataLoader(self.train_set, self.batch_size)
        for e in range(self.epochs):
            with tqdm(total=len(train_loader), bar_format="{l_bar}{bar}{r_bar}{bar:-10b}") as pbar_epoch:
                self.model.train()
                all_loss = []
                for data in train_loader:
                    pbar_epoch.update(1)
                    x, y = data
                    x = x.to(self.device)
                    y = y.to(self.device)
                    self.optimizer.zero_grad()

                    if self.augment_pos == 'input' and self.transform is not None:
                        x = self.transform(x)
                        pred = self.model(x)
                    elif self.augment_pos == 'model':
                        pred = self.model(x, self.transform)
                    elif self.transform is None:
                        pred = self.model(x)
                    else:
                        raise Exception(f'{self.augment_pos} is not expected.')

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
                    val_loss = self.loss_fn(pred, y).item()
                    val_acc = perf_metric('acc', y.cpu().numpy(), pred.cpu().numpy())
                    pbar_epoch.set_postfix({
                        'loss': np.array(all_loss).mean(),
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    })
                    self.writer.add_scalar('val_loss', val_loss, e)
                    self.writer.add_scalar('val_acc', val_acc, e)
                    self.early_stopping(val_loss, self.model)
                    if self.early_stopping.early_stop:
                        print(f'early stopping {e} / {self.epochs}')
                        self.model.load_state_dict(torch.load('checkpoint.pt'))
                        break
