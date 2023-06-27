import os
import time
import math
import numpy as np
from scipy.fftpack import fft
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
from collections import Counter
from utils.DataLoader import SHLDataLoader
from utils.DataProcesser import DataProcesser

from sklearn.model_selection import train_test_split
from utils.Solver import Solver
import seaborn as sn 
from utils.helper import evaluate

# # 一 给数据路径

label_dict = {1: 'Still', 2: 'Walking', 3: 'Run', 4: 'Bike', 5: 'Car', 6: 'Bus', 7: 'Train', 8: 'Subway'}

# # 二 读取特征信息
drop_columns_1  =  ['Unnamed: 0',
 'label_mode',
 'ACC_x_kurtosis',
 'GYR_y_mean',
 'ACC_x_skewness',
 'MAG_x_peak_frequency',
 'distance_bus_stops_change',
 'norm_mag_skewness',
 'ACC_y_skewness',
 'GYR_y_kurtosis',
 'accuracy_pct_change',
 'GYR_z_median',
 'norm_gyr_skewness',
 'speed_change',
 'norm_gyr_kurtosis',
 'GYR_z_mean',
 'MAG_z_peak_frequency',
 'GYR_x_mean',
 'MAG_z_skewness',
 'MAG_y_peak_frequency',
 'MAG_x_kurtosis',
 'GYR_x_median',
 'MAG_x_skewness',
 'MAG_y_kurtosis',
 'GYR_x_skewness',
 'GYR_z_skewness',
 'norm_mag_kurtosis',
 'GYR_y_skewness',
 'MAG_y_skewness',
 'MAG_z_kurtosis'
 ]

drop_columns_2 = ['Unnamed: 0',
 'Unnamed: 0.1',
  'Unnamed: 0.1.1',
 'time_all',
 'time',
 'label_num',
 'time_series1',
 'time_series2',
 'time_y',
 'time_series3',
 'time_z' ,'eid',
 'ACC_x_kurtosis',
 'GYR_y_mean',
 'ACC_x_skewness',
 'MAG_x_peak_frequency',
 'distance_bus_stops_change',
 'norm_mag_skewness',
 'ACC_y_skewness',
 'GYR_y_kurtosis',
 'accuracy_pct_change',
 'GYR_z_median',
 'norm_gyr_skewness',
 'speed_change',
 'norm_gyr_kurtosis',
 'GYR_z_mean',
 'MAG_z_peak_frequency',
 'GYR_x_mean',
 'MAG_z_skewness',
 'MAG_y_peak_frequency',
 'MAG_x_kurtosis',
 'GYR_x_median',
 'MAG_x_skewness',
 'MAG_y_kurtosis',
 'GYR_x_skewness',
 'GYR_z_skewness',
 'norm_mag_kurtosis',
 'GYR_y_skewness',
 'MAG_y_skewness',
 'MAG_z_kurtosis'
 ]


def get_valid_pre_result(data_processed_train_dict,data_processed_validate_dict):
    for k,(train_type,data_processed_train) in enumerate(data_processed_train_dict.items()):
        print('training set:{},the {}th training set'.format(train_type,k))
        print('begin training')
        X_train = data_processed_train.drop(drop_columns_1,axis=1)
        y_train= data_processed_train['label_mode']
        tree_model = Solver(X_train,y_train)
        tree_model.train()
   
        print('begin testing')
        for s,(test_type,data_processed_validate) in enumerate(data_processed_validate_dict.items()):
            print('training set : {} ; testing set : {}; test order: {}'.format(train_type,test_type,s))
            X_test = data_processed_validate.drop(drop_columns_2,axis=1)
            # y_test= data_processed_validate['label_mode']
            tree_model.predict_raw(X_test)
            tree_model.predict_mean()
            # ----------------改成最终确定的那个模型------------------------------------
            y_predict_test = tree_model.pred_mean  
            data_processed_validate['predict_label'] = pd.DataFrame(y_predict_test)
            print('--------------------------predict well----------------------------------------------')
            print('-------------------begin get testing result--------------------------------------')
            data_processed_validate['time_all'] = data_processed_validate['time_all'].apply(lambda x: x.strip('[]'))
            timestamps = data_processed_validate['time_all'].str.split(',').explode().reset_index(drop=True).astype(int)
            labels = data_processed_validate['predict_label'].repeat(data_processed_validate['time_all'].str.split(',').apply(len)).reset_index(drop=True).astype(int)
            test_result_dataframe = pd.DataFrame({'timestamp': timestamps, 'label': labels})
            print('---------------predict all test label generate well-------------------------')
            return test_result_dataframe


# train set  

data_path_train_1 =  '/home/jnli/SHL_2023/SHL2023/data2023/train/Bag'
data_path_train_2 =  '/home/jnli/SHL_2023/SHL2023/data2023/train/Hand'
data_path_train_3 =  '/home/jnli/SHL_2023/SHL2023/data2023/train/Hips'
data_path_train_4 =  '/home/jnli/SHL_2023/SHL2023/data2023/train/Torso'


data_path_valid_1 = '/home/jnli/SHL_2023/SHL2023/data2023/validate/Bag'
data_path_valid_2 = '/home/jnli/SHL_2023/SHL2023/data2023/validate/Hand'
data_path_valid_3 = '/home/jnli/SHL_2023/SHL2023/data2023/validate/Hips'
data_path_valid_4 = '/home/jnli/SHL_2023/SHL2023/data2023/validate/Torso'


data_processed_train_1 = pd.read_csv('{}/get_last_features_balanced_2.csv'.format(data_path_train_1))
data_processed_train_2 = pd.read_csv('{}/get_last_features_balanced_2.csv'.format(data_path_train_2))
data_processed_train_3 = pd.read_csv('{}/get_last_features_balanced_2.csv'.format(data_path_train_3))
data_processed_train_4 = pd.read_csv('{}/get_last_features_balanced_2.csv'.format(data_path_train_4))

data_processed_validate_1 = pd.read_csv('{}/get_last_features_reduce.csv'.format(data_path_valid_1))
data_processed_validate_2 = pd.read_csv('{}/get_last_features_reduce.csv'.format(data_path_valid_2))
data_processed_validate_3 = pd.read_csv('{}/get_last_features_reduce.csv'.format(data_path_valid_3))
data_processed_validate_4 = pd.read_csv('{}/get_last_features_reduce.csv'.format(data_path_valid_4))

# data_processed_train = pd.concat([data_processed_train_1, data_processed_train_2, data_processed_train_3, data_processed_train_4], axis=0)
data_processed_train = pd.concat([data_processed_train_1, data_processed_train_2, data_processed_train_3, data_processed_train_4,data_processed_validate_1, data_processed_validate_2, data_processed_validate_3, data_processed_validate_4], axis=0)
print('the all length of training set:{}'.format(len(data_processed_train)))


# test set
data_path_test = '/home/jnli/SHL_2023/SHL2023/data2023/test'
data_processed_validate = pd.read_csv('{}/get_last_features.csv'.format(data_path_test))

data_processed_train_dict = {'all_train_set':data_processed_train}
data_processed_validate_dict = {'test_set':data_processed_validate}



test_result_dataframe = get_valid_pre_result(data_processed_train_dict,data_processed_validate_dict)
test_result_dataframe.to_csv('{}/Fighting_zsn_predictions_6_25_last.txt'.format(data_path_test), sep='\t', index=False, header=False)
print(' testing result saved well')
print('test over')



