import os
import time

import hydra
from hydra._internal.utils import get_args
from torch.utils.tensorboard import SummaryWriter

from train import SemiTrainer, SupervisedTrainer, XGBoostTrainer
from utils import set_seed


@hydra.main(config_path='conf', config_name='supervised')
def main(config):
    args = get_args()
    print(args.config_name)
    if args.config_name is not None:
        train_method = args.config_name
    else:
        train_method = 'supervised'

    config['data_dir'] = os.path.join(hydra.utils.get_original_cwd(), config['data_dir'])
    writer = SummaryWriter('./')
    set_seed(config['seed'])
    if train_method == 'supervised':
        t = SupervisedTrainer(config, writer)
    elif train_method == 'semi':
        t = SemiTrainer(config, writer)
    elif train_method == 'xgboost':
        t = XGBoostTrainer(config, writer)
    else:
        raise Exception(f'train method\'s {train_method} is unexpected.')
    t.train()
    acc = t.test()
    writer.add_scalar('acc', acc, 1)
    writer.close()


if __name__ == '__main__':
    s = time.time()
    main()
    print(time.time() - s)
