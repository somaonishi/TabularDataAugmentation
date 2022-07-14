import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def categorical2onehot_sklearn(x_train_df, y_train_df, x_test_df, y_test_df):
    num_cal = x_train_df.select_dtypes(include=[int, float]).columns
    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    enc.fit(x_train_df.drop(num_cal, axis=1))

    # onehot encoding for categorical
    x_train_cate = enc.transform(x_train_df.drop(num_cal, axis=1))
    x_test_cate = enc.transform(x_test_df.drop(num_cal, axis=1))

    # categorical + numerical
    x_train = np.concatenate((x_train_df[num_cal].to_numpy(), x_train_cate), axis=1)
    x_test = np.concatenate((x_test_df[num_cal].to_numpy(), x_test_cate), axis=1)
    numerical_idx = [i for i in range(x_train_df[num_cal].to_numpy().shape[1])]
    # label onehot encode
    enc = OneHotEncoder(handle_unknown="ignore")
    y_train = y_train_df.to_numpy().reshape(-1, 1)
    y_test = y_test_df.to_numpy().reshape(-1, 1)
    enc.fit(y_train)
    y_train = enc.transform(y_train).toarray()
    y_test = enc.transform(y_test).toarray()

    return x_train, y_train, x_test, y_test, numerical_idx


def categorical2onehot_pd(x_train_df, x_test_df, y_train_df, y_test_df):
    pass


def remove_missing_feature(df):
    df = df.replace({' ?': None})
    df = df.dropna()
    return df


def mode_missing_feature(df):
    df = df.replace({' ?': None})
    df = df.fillna(df.mode().iloc[0])
    return df
