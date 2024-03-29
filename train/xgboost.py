import logging

from data.data_loader import get_dataset
from utils import perf_metric

import xgboost as xgb

log = logging.getLogger(__name__)


def get_dmatrix(dataset, labeled_rate=1.0):
    x, y = dataset.x, dataset.y
    size = int(len(x) * labeled_rate)
    x, y = x[:size], y[:size]
    y = y.argmax(axis=1).reshape(-1, 1)
    return xgb.DMatrix(x.numpy(), label=y.numpy())


class XGBoostTrainer:
    def __init__(self, config, writer=None) -> None:
        self.writer = writer
        self.train_set, self.val_set, self.test_set = get_dataset(config)
        self.l_dim = self.train_set[0][1].shape[-1]
        self.labeled_rate = config["labeled_rate"]

    def train(self):
        dtrain = get_dmatrix(self.train_set, self.labeled_rate)
        dvalid = get_dmatrix(self.val_set, self.labeled_rate)
        params = {
            'objective': 'multi:softprob',
            'num_class': self.l_dim,
            "eval_metric": "mlogloss"
        }
        self.model = xgb.train(params=params,
                               dtrain=dtrain,
                               evals=[(dtrain, "train"), (dvalid, "valid")],
                               num_boost_round=100,
                               early_stopping_rounds=10,
                               verbose_eval=10)

    def test(self):
        dtest = get_dmatrix(self.test_set)
        pred = self.model.predict(dtest)
        result = perf_metric('acc', self.test_set.y.numpy(), pred)
        log.info(f'Performance: {100 * result}')
        return 100 * result
