import logging

import torch
import torch.nn as nn
import torch.optim as optim
from data.augment import get_transforms
from data.data_loader import get_dataset
from model.model import get_model
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import EarlyStopping, perf_metric

log = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self, config, writer=None) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.writer = writer

        self.train_set, self.val_set, self.test_set = get_dataset(config)
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.test_batch_size = config['test_batch_size']
        self.dim = self.train_set[0][0].shape[-1]
        self.l_dim = self.train_set[0][1].shape[-1]

        self.transform = get_transforms(config['augment'], config['augmenters'])

        self.augment_pos = config['augment_pos']

        self.model = get_model(config['model_name'],
                               self.dim,
                               self.l_dim)
        self.model.to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()

        self.opt_alg = config['opt']
        self.set_optimizer()

        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.99999)
        self.early_stopping = EarlyStopping(patience=config['early_stopping_patience'], path='checkpoint.pt')

    def set_optimizer(self):
        if self.opt_alg == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9)
        elif self.opt_alg == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters())

    def train(self):
        raise NotImplementedError()

    def test(self):
        test_loader = DataLoader(self.test_set, self.test_batch_size)
        results = []
        self.model.eval()
        with tqdm(test_loader, bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}") as pbar_epoch:
            for data in pbar_epoch:
                with torch.no_grad():
                    x, y = data
                    x = x.to(self.device)
                    pred = self.model(x)
                    results.append(perf_metric('acc', y.cpu().numpy(), pred.cpu().numpy()))
        log.info(f'Performance: {100 * torch.tensor(results).mean()}')
        return 100 * torch.tensor(results).mean()
