import itertools
import os
import time

import hydra
from torch.utils.tensorboard import SummaryWriter

from train import Train
from utils import set_seed


@hydra.main(config_path='conf', config_name='config')
def main(config):
    config['data_dir'] = os.path.join(hydra.utils.get_original_cwd(), config['data_dir'])
    writer = SummaryWriter('./')
    set_seed(config['seed'])
    t = Train(config, writer)
    t.train()
    acc = t.test()
    writer.add_scalar('acc', acc, 1)
    writer.close()


if __name__ == '__main__':
    s = time.time()
    main()
    print(time.time() - s)
