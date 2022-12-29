import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random, math, os, time


SEED = 1234
random.seed(SEED)


def generate_samples(x, y, model_params, seq_len_before=7, seq_len_after=7, output_seq_len=9):
  # generate_samples(x.values, y, model_params, 10, 10, 6)
  #Tạo ra mẫu, trả về input_seq: samples gồm các giá trị trước khoản trống, sau khoảng trống và trong khoảng trống =0
  # output_seq: giá trị trong khoản trống

    total_samples = x.shape[0] # 6551
    total_len = seq_len_before + seq_len_after + output_seq_len # 26

    input_batch_idxs = [list(range(i, i + seq_len_before)) + list(
        range(i + seq_len_before + output_seq_len, i + seq_len_before + output_seq_len + seq_len_after)) for i in
                        range((total_samples - total_len + 1))]
 #-> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,   .......,  16, 17, 18, 19, 20, 21, 22, 23, 24, 25]


    input_seq = np.take(x, input_batch_idxs, axis=0) #(20,6) # Lấy ra các values tương ứng với index trên

    z = np.zeros((output_seq_len, model_params['dim_in'])) #(6,6) giá trị điền khuyết

    input_seq = np.array([np.concatenate((i[:seq_len_before], z, i[seq_len_before:])) for i in input_seq])
    # 6526, 26, 6 nối lại khỏi tạo khúc giữa = 0 

    output_batch_idxs = [list(range(i + seq_len_before, i + seq_len_before + output_seq_len)) for i in
                         range((total_samples - total_len + 1))]  # len = 6256
    # [10, 11, 12, 13, 14, 15] index khuyết 
    output_seq = np.take(y, output_batch_idxs, axis=0) #(6526, 6) giá trị bị lấy ra 

    return input_seq, output_seq #(6526, 26, 6), (6526, 6)


def pad_all_cases(x, y, model_params, min_len_before=7, max_len_before=9, min_len_after=7, max_len_after=9, targetlength=9):

#(df_loc, df_loc['NO3'], train_sampling_params,   min_len_before=10, max_len_before=10,
# min_len_after=10, max_len_after=10, targetlength=6 )

    total_x = []
    total_y = []
    total_len_x = []
    totle_len_before_x = []
    
    #for l_before in range(10, 11):
      #for l_after in range(10, 11):


    for l_before in range(min_len_before, max_len_before + 1):
        for l_after in range(min_len_after, max_len_after + 1):
            case_x, case_y = generate_samples(x.values, y, model_params, l_before, l_after, targetlength)
            # npad là 1 tuple của (n_before, n_after) cho mỗi chiều

            len_x = np.full(case_x.shape[0], case_x.shape[1]) #(6526,) [26,26,26,26,.....26,26,26]
            #Trả về một mảng mới có hình dạng và loại đã cho, chứa đầy fill_value.

            len_before_sequence_x = np.full(case_x.shape[0], l_before) #(6526,) [10,10,10,....10,10]

            npad = ((0, 0), (0, max_len_before - l_before + max_len_after - l_after), (0, 0)) #((0, 0), (0, 0), (0, 0))


            same_length_x = np.pad(case_x, pad_width=npad, mode='constant', constant_values=0)

            total_x.append(same_length_x) # len total_x = 1 
            total_y.append(case_y)
            total_len_x.append(len_x)
            totle_len_before_x.append(len_before_sequence_x)

    ## total x,y
    concatenated_x = np.concatenate(total_x, axis=0) #(6526, 26, 6)
    concatenated_y = np.concatenate(total_y, axis=0) #(6526, 6)
    len_all_case = np.concatenate(total_len_x).ravel()  #(6526,)
    len_before_all_case = np.concatenate(totle_len_before_x).ravel()  #(6526,)

    print('concatenated_x', concatenated_x.shape)
    print('concatenated_y', concatenated_y.shape)
    print('len_all_case', len_all_case.shape)
    print('len_before_all_case', len_before_all_case.shape)

    return concatenated_x, concatenated_y, len_all_case, len_before_all_case
