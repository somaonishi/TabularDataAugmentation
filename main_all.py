import itertools
import logging
import os
import time

import hydra
from hydra._internal.utils import get_args
from torch.utils.tensorboard import SummaryWriter

from train import SemiTrainer, SupervisedTrainer
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
    ori_augmenters = config['augmenters']
    cnt = 0
    max_acc = 0
    max_comb = []
    for i in range(6):
        for augmenters in itertools.combinations(config['augmenters'], i):
            augmenters = list(augmenters)
            logging.info(f'[{cnt}] Augmenters: {augmenters}')
            config['augmenters'] = augmenters
            set_seed(config['seed'])

            if train_method == 'supervised':
                t = SupervisedTrainer(config, writer)
            elif train_method == 'semi':
                t = SemiTrainer(config, writer)
            else:
                raise Exception(f'train method\'s {train_method} is unexpected.')

            t.train()
            acc = t.test()
            writer.add_scalar('acc', acc, cnt)
            if acc >= max_acc:
                if acc == max_acc:
                    max_comb.append(augmenters)
                else:
                    max_acc = acc
                    max_comb = augmenters
            cnt += 1
            config['augmenters'] = ori_augmenters
    writer.close()
    print(max_acc.item())
    print(max_comb)


if __name__ == '__main__':
    s = time.time()
    main()
    print(time.time() - s)
