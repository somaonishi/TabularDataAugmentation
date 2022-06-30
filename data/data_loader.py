import os
from typing import Union

import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.datasets import load_boston, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .dataset import TabularDataset
from .utils import (categorical2onehot_sklearn, mode_missing_feature,
                    remove_missing_feature)

income_columns = ['age', 'workclass', 'fnlwgt', 'education',
                  'education-num', 'marital-status', 'occupation',
                  'relationship', 'race', 'sex', 'capital-gain',
                  'capital-loss', 'hours-per-week', 'native-country', 'income']


def read_csv(path, label, columns, missing_fn=None, header=None) -> Union[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, header=header, names=columns)
    if missing_fn is not None:
        df = missing_fn(df)
    y = df.pop(label)
    return df, y


def mnist_to_tabular(x, y):
    x = x / 255.0
    y = np.asarray(pd.get_dummies(y))

    # flatten
    no, dim_x, dim_y = np.shape(x)
    x = np.reshape(x, [no, dim_x * dim_y])
    return x, y


def get_dataset(config):
    '''
    input:
        data_name: データの名前
        label_data_rate: ラベルデータの割合
    return:
        labeled dataset, unlabeled dataset, test dataset
        unlabeled datasetはラベルを含むデータセットだが学習では使わない
    '''
    data_name = config['data_name']
    data_dir = config['data_dir']
    scalar_name = config['scalar']
    cate_num = 0

    if data_name == 'iris':
        data = load_iris()
        data.target = np.asarray(pd.get_dummies(data.target))
        x_train, x_test, y_train, y_test = train_test_split(
            data.data,
            data.target,
            test_size=0.2,
            random_state=42)
    elif data_name == 'wine':
        data = load_wine()
        data.target = np.asarray(pd.get_dummies(data.target))
        x_train, x_test, y_train, y_test = train_test_split(
            data.data,
            data.target,
            test_size=0.2,
            random_state=42)
    elif data_name == 'boston':
        data = load_boston()
        data.target = np.asarray(pd.get_dummies(data.target))
        x_train, x_test, y_train, y_test = train_test_split(
            data.data,
            data.target,
            test_size=0.2,
            random_state=42)
    elif data_name == 'mnist':
        train_set = torchvision.datasets.MNIST(data_dir, train=True, download=True)
        test_set = torchvision.datasets.MNIST(data_dir, train=False, download=True)
        x_train, y_train = mnist_to_tabular(train_set.data.numpy(), train_set.targets.numpy())
        x_test, y_test = mnist_to_tabular(test_set.data.numpy(), test_set.targets.numpy())
    elif data_name == 'income':
        missing_fn = remove_missing_feature
        missing_fn = mode_missing_feature
        missing_fn = None
        x_train_df, y_train_df = read_csv(os.path.join(data_dir, 'income/train.csv'), 'income', income_columns, missing_fn=missing_fn)
        x_test_df, y_test_df = read_csv(os.path.join(data_dir, 'income/test.csv'), 'income', income_columns, missing_fn=missing_fn)
        x_train_df.pop('education-num')
        x_test_df.pop('education-num')
        # " <=50k.", ">50k." to " <=50k", ">50k"
        y_test_df = pd.DataFrame([s.rstrip('.') for s in y_test_df.values])
        x_train, y_train, x_test, y_test, cate_num = categorical2onehot_sklearn(x_train_df, y_train_df, x_test_df, y_test_df)

    scalar = None
    if scalar_name == 'minmax':
        scalar = MinMaxScaler()
    elif scalar_name == 'standard':
        scalar = StandardScaler()

    if scalar is not None:
        scalar.fit(x_train[:, cate_num:])
        x_train[:, cate_num:] = scalar.transform(x_train[:, cate_num:])
        x_test[:, cate_num:] = scalar.transform(x_test[:, cate_num:])

    idx = np.random.permutation(len(x_train))
    train_idx = idx[:int(len(idx) * 0.9)]
    valid_idx = idx[int(len(idx) * 0.9):]

    x_val = x_train[valid_idx]
    y_val = y_train[valid_idx]
    x_train = x_train[train_idx]
    y_train = y_train[train_idx]

    return TabularDataset(x_train, y_train),\
        TabularDataset(x_val, y_val),\
        TabularDataset(x_test, y_test)


if __name__ == '__main__':
    _, _, a = get_dataset('income', 0.1)
    print(a)
