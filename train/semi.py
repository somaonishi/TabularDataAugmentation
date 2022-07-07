import numpy as np
import torch
import torch.nn as nn
from data.augment import get_transforms
from data.dataset import SemiDataset, TabularDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import perf_metric

from .base import BaseTrainer
from .losses import FixMatchLoss, NormalLoss


def to_unlabel(train_set: TabularDataset, val_set: TabularDataset, l_rate=0.1):
    x_train, y_train = train_set.x.numpy(), train_set.y.numpy()
    x_val, y_val = val_set.x.numpy(), val_set.y.numpy()
    x = np.concatenate([x_train, x_val])
    y = np.concatenate([y_train, y_val])

    label_size = int(len(x) * l_rate)
    l_x, u_x = x[:label_size], x[label_size:]
    l_y = y[:label_size]

    idx = np.random.permutation(len(l_x))
    train_idx = idx[:int(len(idx) * 0.9)]
    valid_idx = idx[int(len(idx) * 0.9):]

    x_val = l_x[valid_idx]
    y_val = l_y[valid_idx]
    x_train = l_x[train_idx]
    y_train = l_y[train_idx]

    return SemiDataset(x_train, y_train, u_x),\
        TabularDataset(x_val, y_val)


class SemiTrainer(BaseTrainer):
    def __init__(self, config, writer=None) -> None:
        super().__init__(config, writer)
        self.labeled_rate = config['labeled_rate']
        self.semi_method = config['method']
        self.semi_para = config['semi_para'][self.semi_method]
        self.beta = config['beta']

        if self.semi_method == 'normal':
            self.u_loss_fn = NormalLoss()
        elif self.semi_method == 'fixmatch':
            self.u_loss_fn = FixMatchLoss()
            self.strong_aug = get_transforms(self.semi_para.strong, config['augmenters'])
        else:
            raise Exception(f'{self.semi_method} is unexpected values')

        self.train_set, self.val_set = to_unlabel(self.train_set,
                                                  self.val_set,
                                                  self.labeled_rate)

    def training_pred(self, x, transform=None) -> torch.Tensor:
        if transform is None:
            transform = self.transform

        if self.augment_pos == 'input' and transform is not None:
            x = transform(x)
            pred = self.model(x)
        elif self.augment_pos == 'model':
            pred = self.model(x, transform)
        elif transform is None:
            pred = self.model(x)
        else:
            raise Exception(f'{self.augment_pos} is not expected.')
        return pred

    def loss_normal_semi(self, u_x):
        u_x_k = torch.cat([u_x for _ in range(self.semi_para.k)])
        u_pred_k = self.training_pred(u_x_k)
        u_pred = u_pred_k.view(self.semi_para.k, len(u_x), -1)
        u_loss = self.u_loss_fn(u_pred)
        return u_loss

    def loss_fixmatch(self, u_x):
        weak = self.training_pred(u_x, self.transform)
        strong = self.training_pred(u_x, self.strong_aug)
        u_loss = self.u_loss_fn(weak, strong)
        return u_loss

    def cal_loss(self, data):
        l_x, l_y, u_x = data
        l_x = l_x.to(self.device)
        l_y = l_y.to(self.device)
        u_x = u_x.to(self.device)

        l_pred = self.training_pred(l_x)
        l_loss = self.loss_fn(l_pred, l_y)

        if self.semi_method == 'normal':
            u_loss = self.loss_normal_semi(u_x)
        elif self.semi_method == 'fixmatch':
            u_loss = self.loss_fixmatch(u_x)

        loss = l_loss + self.beta * u_loss
        return loss, l_loss, u_loss

    def train(self):
        self.semi_train()

    def semi_train(self):
        train_loader = DataLoader(self.train_set, self.batch_size, shuffle=True)
        for e in range(self.epochs):
            with tqdm(total=len(train_loader), bar_format="{l_bar}{bar}{r_bar}{bar:-10b}") as pbar_epoch:
                self.model.train()
                all_loss = []
                all_l_loss = []
                all_u_loss = []
                for data in train_loader:
                    pbar_epoch.update(1)
                    self.optimizer.zero_grad()
                    loss, l_loss, u_loss = self.cal_loss(data)
                    loss.backward()
                    self.optimizer.step()

                    all_loss.append(loss.item())
                    all_l_loss.append(l_loss.item())
                    all_u_loss.append(u_loss.item())
                    pbar_epoch.set_description(f"epoch[{e + 1} / {self.epochs}]")
                    pbar_epoch.set_postfix({'loss': loss.item(),
                                            'l_loss': l_loss.item(),
                                            'u_loss': u_loss.item()})
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
                        'l_loss': np.array(all_l_loss).mean(),
                        'u_loss': np.array(all_u_loss).mean(),
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
