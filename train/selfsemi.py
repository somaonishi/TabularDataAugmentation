import torch.nn as nn
import torch.optim as optim
from model.selfsemi import AE, MLP

from .semi import SemiTrainer


class SelfSemiTrainer(SemiTrainer):
    def __init__(self, config, writer=None) -> None:
        super().__init__(config, writer)
        self.ae = AE(self.dim, self.dim).to(self.device)
        self.self_lossfn = nn.MSELoss()
        if config['self_opt'] == 'RMSprop':
            self.self_optimizer = optim.RMSprop(self.ae.parameters(), lr=1e-3)
        elif config['self_opt'] == 'Adam':
            self.self_optimizer = optim.Adam(self.ae.parameters())

    def self_train(self):
        pass

    def train(self):
        self.self_train()
        self.model = MLP(self.dim, self.l_dim, self.ae.encoder).to(self.device)
        self.set_optimizer()
        self.semi_train()
