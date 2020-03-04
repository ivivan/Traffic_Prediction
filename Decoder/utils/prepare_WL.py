import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import random
import math
import os
import time

from utils.VLSW import pad_all_cases

# set the random seeds for reproducability
SEED = 1234
random.seed(SEED)


def preprocess_df(df):
    """ The training and testing data are manually selected.
    :param df:  dataframe with raw data
    :return:
    """

    df.set_index('date', inplace=True)

    WL = df['wl'].values.copy().reshape(-1, 1)

    # Standlization, use StandardScaler
    scaler_x = MinMaxScaler()
    scaler_x.fit(df[['wl']])
    df[['wl']] = scaler_x.transform(
            df[['wl']])

    scaler_y = MinMaxScaler()
    scaler_y.fit(WL)
    y_all = scaler_y.transform(WL)

    df_train = df.loc['19/11/2017 0:00':'31/08/2019 23:50'].copy()
    df_test = df.loc['1/09/2019 0:00':'25/11/2019 23:50'].copy()

    return df_train, df_test, scaler_x, scaler_y


def train_val_test_generate(dataframe, model_params):
    '''
    :param dataframe: processed dataframe
    :param model_params: for input dim
    :return: train_x, train_y, test_x, test_y with the same length (by padding zero)
    '''

    train_val_test_x, train_val_test_y, len_x_samples, len_before_x_samples = pad_all_cases(
        dataframe, dataframe['wl'].values, model_params,
        model_params['min_before'], model_params['max_before'],
        model_params['min_after'], model_params['max_after'],
        model_params['output_length'])

    train_val_test_y = np.expand_dims(train_val_test_y, axis=2)

    return train_val_test_x, train_val_test_y, len_x_samples, len_before_x_samples


def train_test_split_SSIM(x, y, x_len, x_before_len, model_params, SEED):
    '''
    :param x: all x samples
    :param y: all y samples
    :param model_params: parameters
    :param SEED: random SEED
    :return: train set, test set
    '''

    # check and remove samples with NaN (just incase)
    index_list = []
    for index, (x_s, y_s, len_s,
                len_before_s) in enumerate(zip(x, y, x_len, x_before_len)):
        if (np.isnan(x_s).any()) or (np.isnan(y_s).any()):
            index_list.append(index)

    x = np.delete(x, index_list, axis=0)
    y = np.delete(y, index_list, axis=0)
    x_len = np.delete(x_len, index_list, axis=0)
    x_before_len = np.delete(x_before_len, index_list, axis=0)

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=None,
                                                        random_state=SEED,
                                                        shuffle=False)

    x_train_len, x_test_len = train_test_split(x_len,
                                               test_size=None,
                                               random_state=SEED,
                                               shuffle=False)

    x_train_before_len, x_test_before_len = train_test_split(x_before_len,
                                                             test_size=None,
                                                             random_state=SEED,
                                                             shuffle=False)

    return x_train, y_train, x_train_len, x_train_before_len


def test_pm25_single_station():
    train_sampling_params = {
        'dim_in': 1,
        'output_length': 6,
        'min_before': 72,
        'max_before': 72,
        'min_after': 0,
        'max_after': 0,
        'file_path': '../data/simplified_PM25.csv'
    }

    test_sampling_params = {
        'dim_in': 1,
        'output_length': 6,
        'min_before': 72,
        'max_before': 72,
        'min_after': 0,
        'max_after': 0,
        'file_path': '../data/simplified_PM25.csv'
    }

    filepath = './data/WL.csv'
    # filepath = '/OSM/CBR/AF_WQ/source/ML/Pytorch_Seq2Seq/Data/WL.csv'


    df = pd.read_csv(filepath, dayfirst=True)

    df_train, df_test, scaler_x, scaler_y = preprocess_df(df)

    x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
        df_train, train_sampling_params)

    print('X_samples:{}'.format(x_samples.shape))
    print('y_samples:{}'.format(y_samples.shape))

    x_train, y_train, x_train_len, x_train_before_len = train_test_split_SSIM(
        x_samples, y_samples, x_len, x_before_len, train_sampling_params, SEED)

    print('x_train:{}'.format(x_train.shape))
    print('y_train:{}'.format(y_train.shape))
    print('x_train_len:{}'.format(x_train_len.shape))
    print('x_train_before_len:{}'.format(x_train_before_len.shape))


    x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
        df_test, test_sampling_params)

    print('X_samples:{}'.format(x_samples.shape))
    print('y_samples:{}'.format(y_samples.shape))

    x_test, y_test, x_test_len, x_test_before_len = train_test_split_SSIM(
        x_samples, y_samples, x_len, x_before_len, test_sampling_params, SEED)

    print('x_test:{}'.format(x_test.shape))
    print('y_test:{}'.format(y_test.shape))
    print('x_test_len:{}'.format(x_test_len.shape))
    print('x_test_before_len:{}'.format(x_test_before_len.shape))

    return (x_train, y_train, x_train_len,
            x_train_before_len), (x_test, y_test, x_test_len,
                                  x_test_before_len), (scaler_x, scaler_y)


if __name__ == "__main__":
    test_pm25_single_station()
    # train_sampling_params = {
    #     'dim_in': 11,
    #     'output_length': 5,
    #     'min_before': 5,
    #     'max_before': 5,
    #     'min_after': 5,
    #     'max_after': 5,
    #     'file_path': '../data/simplified_PM25.csv'
    # }

    # test_sampling_params = {
    #     'dim_in': 11,
    #     'output_length': 5,
    #     'min_before': 5,
    #     'max_before': 5,
    #     'min_after': 5,
    #     'max_after': 5,
    #     'file_path': '../data/simplified_PM25.csv'
    # }

    # filepath = 'data/simplified_PM25.csv'
    # df = pd.read_csv(filepath, dayfirst=True)

    # df_train, df_test, y, scaler_x, scaler_y = preprocess_df(df)

    # x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
    #     df_train, train_sampling_params)

    # print('X_samples:{}'.format(x_samples.shape))
    # print('y_samples:{}'.format(y_samples.shape))

    # x_train, y_train, x_train_len, x_train_before_len = train_test_split_SSIM(
    #     x_samples, y_samples, x_len, x_before_len, train_sampling_params, SEED)

    # print('x_train:{}'.format(x_train.shape))
    # print('y_train:{}'.format(y_train.shape))
    # print('x_train_len:{}'.format(x_train_len.shape))
    # print('x_train_before_len:{}'.format(x_train_before_len.shape))

    # x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
    #     df_test, test_sampling_params)

    # print('X_samples:{}'.format(x_samples.shape))
    # print('y_samples:{}'.format(y_samples.shape))

    # x_test, y_test, x_test_len, x_test_before_len = train_test_split_SSIM(
    #     x_samples, y_samples, x_len, x_before_len, test_sampling_params, SEED)

    # print('x_test:{}'.format(x_test.shape))
    # print('y_test:{}'.format(y_test.shape))
    # print('x_test_len:{}'.format(x_test_len.shape))
    # print('x_test_before_len:{}'.format(x_test_before_len.shape))

    # print('split train/test array')
    # x_test_list = np.split(x_test, [5, 10], axis=1)
    # x_train_list = np.split(x_train, [5, 10], axis=1)

    # for i in x_test_list:
    #     print(i.shape)
