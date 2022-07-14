import os
from typing import Union

import numpy as np
import pandas as pd
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


def read_csv(path, label=None, columns=None, missing_fn=None, header=None) -> Union[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, header=header, names=columns)
    if missing_fn is not None:
        df = missing_fn(df)
    if label is None:
        label = len(df.columns) - 1
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

    if data_name == 'iris':
        data = load_iris()
        data.target = np.asarray(pd.get_dummies(data.target))
        x_train, x_test, y_train, y_test = train_test_split(
            data.data,
            data.target,
            test_size=0.2,
            random_state=42)
        numerical_idx = [i for i in range(x_train.shape[1])]
    elif data_name == 'wine':
        data = load_wine()
        data.target = np.asarray(pd.get_dummies(data.target))
        x_train, x_test, y_train, y_test = train_test_split(
            data.data,
            data.target,
            test_size=0.2,
            random_state=42)
        numerical_idx = [i for i in range(x_train.shape[1])]
    elif data_name == 'boston':
        data = load_boston()
        data.target = np.asarray(pd.get_dummies(data.target))
        x_train, x_test, y_train, y_test = train_test_split(
            data.data,
            data.target,
            test_size=0.2,
            random_state=42)
        numerical_idx = [i for i in range(x_train.shape[1])]
    elif data_name == 'mnist':
        train_set = torchvision.datasets.MNIST(data_dir, train=True, download=True)
        test_set = torchvision.datasets.MNIST(data_dir, train=False, download=True)
        x_train, y_train = mnist_to_tabular(train_set.data.numpy(), train_set.targets.numpy())
        x_test, y_test = mnist_to_tabular(test_set.data.numpy(), test_set.targets.numpy())
        numerical_idx = [i for i in range(x_train.shape[1])]
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
        x_train, y_train, x_test, y_test, numerical_idx = categorical2onehot_sklearn(x_train_df, y_train_df, x_test_df, y_test_df)
    elif data_name == 'covertype':
        x_df, y_df = read_csv(os.path.join(data_dir, 'covertype/covertype.csv'))
        x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=42)
        x_train = x_train.to_numpy()
        x_test = x_test.to_numpy()
        y_train = np.identity(8)[y_train]
        y_test = np.identity(8)[y_test]
        y_train = np.delete(y_train, 0, 1)
        y_test = np.delete(y_test, 0, 1)
        numerical_idx = [i for i in range(x_train.shape[1])]
    elif data_name == 'blog':
        # Label data is in column 280
        x_train, y_train = read_csv(os.path.join(data_dir, 'blog/blogData_train.csv'), 280, columns=None, missing_fn=None)
        x_test, y_test = read_csv(os.path.join(data_dir, 'blog/blogData_test.csv'), 280, columns=None, missing_fn=None)
        print(len(x_test) / (len(x_test) + len(x_train)))
        x_train = x_train.to_numpy()
        x_test = x_test.to_numpy()
        y_train = np.sign(y_train)
        y_train = y_train.astype(int)
        y_train = np.identity(2)[y_train]
        y_test = np.sign(y_test)
        y_test = y_test.astype(int)
        y_test = np.identity(2)[y_test]
        cate_idx = [i for i in range(62, 276)]
        numerical_idx = [i for i in range(x_train.shape[1]) if i not in cate_idx]

    scalar = None
    if scalar_name == 'minmax':
        scalar = MinMaxScaler()
    elif scalar_name == 'standard':
        scalar = StandardScaler()

    if scalar is not None:
        scalar.fit(x_train[:, numerical_idx])
        x_train[:, numerical_idx] = scalar.transform(x_train[:, numerical_idx])
        x_test[:, numerical_idx] = scalar.transform(x_test[:, numerical_idx])

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
