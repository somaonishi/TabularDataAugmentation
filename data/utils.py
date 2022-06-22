import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def categorical2onehot_sklearn(x_train_df, y_train_df, x_test_df, y_test_df):
    num_cal = x_train_df.select_dtypes(include=[int, float]).columns
    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    enc.fit(x_train_df.drop(num_cal, axis=1))

    # onehot encoding for categorical
    x_train = enc.transform(x_train_df.drop(num_cal, axis=1))
    x_test = enc.transform(x_test_df.drop(num_cal, axis=1))

    # categorical + numerical
    cate_num = x_train.shape[1]
    x_train = np.concatenate((x_train, x_train_df[num_cal].to_numpy()), axis=1)
    x_test = np.concatenate((x_test, x_test_df[num_cal].to_numpy()), axis=1)
    # label onehot encode
    enc = OneHotEncoder(handle_unknown="ignore")
    y_train = y_train_df.to_numpy().reshape(-1, 1)
    y_test = y_test_df.to_numpy().reshape(-1, 1)
    enc.fit(y_train)
    y_train = enc.transform(y_train).toarray()
    y_test = enc.transform(y_test).toarray()

    return x_train, y_train, x_test, y_test, cate_num


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
