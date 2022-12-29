import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import random, math, os, time

from utils.VLSW import pad_all_cases


SEED = 1234
random.seed(SEED)


def series_to_superviesed(x_timeseries, y_timeseries, n_memory_step, n_forcast_step, split=None):
    '''
        x_timeseries: giá trị time series vào, numpy array, (time_step, features)
        y_timeseries: giá trị target time series,  numpy array, (time_step, features)
        n_memory_step: Số bước nhớ trong học có giám sát, int
        n_forcast_step: số bước dự đoán, int
        split: Phần được sử dụng làm tập huấn luyện, float, vd: 0.8 - tỷ lệ chia tập train
    '''
    assert len(x_timeseries.shape) == 2, 'x_timeseries phải có dạng (time_step, features)'
    assert len(y_timeseries.shape) == 2, 'y_timeseries phải có dạng (time_step, features)'

    input_step, input_feature = x_timeseries.shape
    output_step, output_feature = y_timeseries.shape
    assert input_step == output_step, 'số time_step của x_timeseries và y_timeseries không bằng nhau!'

    n_RNN_sample = input_step - n_forcast_step - n_memory_step + 1
    RNN_x = np.zeros((n_RNN_sample, n_memory_step, input_feature))
    RNN_y = np.zeros((n_RNN_sample, n_forcast_step, output_feature))

    for n in range(n_RNN_sample):
        RNN_x[n, :, :] = x_timeseries[n:n + n_memory_step, :]
        RNN_y[n, :, :] = y_timeseries[n + n_memory_step:n + n_memory_step +
                                      n_forcast_step, :]
    if split != None:
        assert (split <= 0.9) & (split >= 0.1), 'Phân chia không hợp lý'
        return RNN_x[:int(split * len(RNN_x))], RNN_y[:int(split * len(RNN_x))], \
               RNN_x[int(split * len(RNN_x)) + 1:], RNN_y[int(split * len(RNN_x)) + 1:]
    else:
        return RNN_x, RNN_y, None, None


def preprocess_df(df):
    """ Tạo ra dữ liệu train và test
    :param df:  dataframe với dữ liệu thô
    :return:
    """

    df.set_index('Timestamp', inplace=True)

    ## Xóa đi các thuộc tính k cần thiết
    df.drop(['Dayofweek'], axis=1, inplace=True)
    df.drop(['Month'], axis=1, inplace=True)

    tw = df['NO3'].values.copy().reshape(-1, 1)
    #df['NO3]: (29052,) -> tw: (29052,1)

    # Standlization, dùng MinMaxScaler
    scaler_x = MinMaxScaler()
    scaler_x.fit(df[['Q', 'Conductivity', 'NO3', 'Temp', 'Turbidity','Level']])

    df[['Q', 'Conductivity', 'NO3', 'Temp', 'Turbidity','Level']] = scaler_x.transform(df[['Q', 'Conductivity', 'NO3', 'Temp', 'Turbidity','Level']])
    #-> MinMaxScaler(df)

    scaler_y = MinMaxScaler()
    scaler_y.fit(tw)
    y_all = scaler_y.transform(tw)

    df_train_one = df.loc['2019-01-01T00:00':'2019-09-30T23:00'].copy() #(6551,6)
    df_test_one = df.loc['2019-10-01T00:00':'2019-12-31T23:00'].copy() #(2207,6)
   
    return df_train_one, df_test_one, scaler_x, scaler_y
    # (6651,6)             (2207,6)    MinmaxScaler   MinmaxScaler




def train_val_test_generate(dataframe, model_params):
    '''
    :param dataframe: processed dataframe
    :param model_params: các giá trị input
    :return: train_x, train_y, test_x, test_y với cùng kích thước (bằng cách pad 0)
    '''

    train_val_test_x, train_val_test_y, len_x_samples, len_before_x_samples = pad_all_cases(
        dataframe, dataframe['NO3'].values, model_params,
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
    :return: tập train, tập test
    '''

    # Kiểm tra và xóa các mẫu NaN
    index_list = []
    for index, (x_s, y_s, len_s, len_before_s) in enumerate(zip(x, y, x_len, x_before_len)):
        if (np.isnan(x_s).any()) or (np.isnan(y_s).any()):
            index_list.append(index)

    x = np.delete(x, index_list, axis=0)
    y = np.delete(y, index_list, axis=0)
    x_len = np.delete(x_len, index_list, axis=0)
    x_before_len = np.delete(x_before_len, index_list, axis=0)

    print('x:{}'.format(x.shape))
    print('y:{}'.format(y.shape))

    return x, y, x_len, x_before_len


def test_qld_single_station(len_b, len_a, graphs):
    train_sampling_params = {
        'dim_in': 6,
        'output_length': graphs,
        'min_before': len_b,
        'max_before': len_b,
        'min_after': len_a,
        'max_after': len_a,
        'file_path': '/content/gdrive/MyDrive/Project_HK1_2022/Data_Visualization/data/data_Mực_Nước/Mulgrave_nomiss.csv'
    }

    test_sampling_params = {
        'dim_in': 6,
        'output_length': graphs,
        'min_before': len_b,
        'max_before': len_b,
        'min_after': len_a,
        'max_after': len_a,
        'file_path': '/content/gdrive/MyDrive/Project_HK1_2022/Data_Visualization/data/data_Mực_Nước/Mulgrave_nomiss.csv'
    }

    filepath = '/content/gdrive/MyDrive/Project_HK1_2022/Data_Visualization/data/data_Mực_Nước/Mulgrave_nomiss.csv'

    df = pd.read_csv(filepath)

    df_train_one, df_test_one, scaler_x, scaler_y = preprocess_df(df)

    print('train_preprocess:{}'.format(df_train_one.shape))
    print('test_preprocess:{}'.format(df_test_one.shape))


    # generate train/test samples seperately

    # train 1
    x_samples, y_samples, x_len, x_before_len = train_val_test_generate( df_train_one, train_sampling_params)

    x_train_one, y_train_one, x_train_len_one, x_train_before_len_one = train_test_split_SSIM(
        x_samples, y_samples, x_len, x_before_len, train_sampling_params, SEED)

  

    x_train = x_train_one
    y_train = y_train_one

    #------------------------------#

    # test 1

    x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
        df_test_one, test_sampling_params)

    x_test_one, y_test_one, x_test_len_one, x_test_before_len_one = train_test_split_SSIM(
        x_samples, y_samples, x_len, x_before_len, test_sampling_params, SEED)



    x_test = x_test_one
    y_test = y_test_one

    print('x_train:{}'.format(x_train.shape))
    print('y_train:{}'.format(y_train.shape))
    print('x_test:{}'.format(x_test.shape))
    print('y_test:{}'.format(y_test.shape))

    print('split train/test array')
    x_test_list = np.split(x_test, [len_b, len_b+graphs], axis=1)
    x_train_list = np.split(x_train, [len_b, len_b+graphs], axis=1)

    for i in x_test_list:
        print(i.shape)

    return (x_train, y_train), (x_test, y_test), (scaler_x, scaler_y)


if __name__ == "__main__":
    len_a, len_b, graphs = 10,10,6
    _,_,_ = test_qld_single_station(len_a, len_b, graphs)
