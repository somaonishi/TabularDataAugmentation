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
    ori_augmenters = config['augmenters']
    cnt = 0
    max_acc = 0
    max_comb = []
    for i in range(6):
        for augmenters in itertools.combinations(config['augmenters'], i):
            augmenters = list(augmenters)
            print(augmenters)
            config['augmenters'] = augmenters
            set_seed(config['seed'])
            t = Train(config, writer)
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
