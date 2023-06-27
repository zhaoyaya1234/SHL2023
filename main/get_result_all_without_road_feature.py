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

drop_columns  =  ['Unnamed: 0',
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
 'MAG_z_kurtosis',
 'error',
 'code',
 'distance_London',
 'distance_Brighton',
 'distance_parks',
 'distance_bus_stops',
 'distance_bus_routes',
 'distance_subway',
 'distance_railways',
 'distance_London_change',
 'distance_Brighton_change',
 'distance_parks_change',
 'distance_bus_routes_change',
 'distance_subway_change',
 'distance_railways_change',
 'error_change'
 ]

label_dict = {1: 'Still', 2: 'Walking', 3: 'Run', 4: 'Bike', 5: 'Car', 6: 'Bus', 7: 'Train', 8: 'Subway'}

features_importance_all = 0

# 二 读取特征信息

def get_valid_pre_result(data_processed_train_dict,data_processed_validate_dict):
    global features_importance_all
    for k,(train_type,data_processed_train) in enumerate(data_processed_train_dict.items()):
        print('training set:{},the {}th training set'.format(train_type,k))
        print('begin training')
        X_train = data_processed_train.drop(drop_columns,axis=1)
        y_train= data_processed_train['label_mode']
        tree_model = Solver(X_train,y_train)
        tree_model.train()
   
        print('begin testing')
        for s,(test_type,data_processed_validate) in enumerate(data_processed_validate_dict.items()):
            print('training set : {} ; testing set : {}; test order: {}'.format(train_type,test_type,s))
            
           
            X_test = data_processed_validate.drop(drop_columns,axis=1)
            
            y_test= data_processed_validate['label_mode']
            tree_model.predict_raw(X_test,y_test)
            tree_model.predict_mean()
            print('training set : {} ; testing set : {}  testing result'.format(train_type,test_type))
            print('tree_model.pred_mean')
            evaluate(y_test, tree_model.pred_mean, normalize = True, names = list(label_dict.values()))
            print('test over')

            print('get importance point')
            tree_model.calculate_feature_importance()
            features_importance_all +=tree_model.feature_importance
            print('importance feature add well')


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

print('1 --------------------- 原始训练集当训练集,原始验证集当测试集的实验---------------------') 
data_processed_train = pd.concat([data_processed_train_1, data_processed_train_2, data_processed_train_3, data_processed_train_4], axis=0)
print('the all length of training set:{}'.format(len(data_processed_train)))
data_processed_validate  = pd.concat([data_processed_validate_1, data_processed_validate_2, data_processed_validate_3, data_processed_validate_4], axis=0)
print('the all length of test set:{}'.format(len(data_processed_validate)))
data_processed_train_dict = {'all_train_set':data_processed_train}

print('begin training all')
data_processed_validate_dict = {'all_test_set':data_processed_validate,'Torso':data_processed_validate_4,'Bag':data_processed_validate_1,'Hand':data_processed_validate_2,'Hips':data_processed_validate_3}
get_valid_pre_result(data_processed_train_dict,data_processed_validate_dict)



print('Obtain the overall feature importance score')
column_names = list(data_processed_validate_1.drop(drop_columns,axis=1).columns)
# Create a dictionary mapping feature names to importance values
feature_importance_dict = {column_names[i]: features_importance_all[i] for i in range(len(column_names))}
# Sort the dictionary by importance values in descending order
feature_importance_dict = dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True))
np.save('feature_importance_dict.npy',feature_importance_dict) # 注意带上后缀名
print('1 feature_importance save well')

# feature_importance_dict= np.load('feature_importance_dict.npy',allow_pickle=True).item()

data_processed_train_dict = {'Hips':data_processed_train_3,'Bag':data_processed_train_1,'Hand':data_processed_train_2,'Torso':data_processed_train_4}
data_processed_validate_dict = {'Torso':data_processed_validate_4,'Bag':data_processed_validate_1,'Hand':data_processed_validate_2,'Hips':data_processed_validate_3}
get_valid_pre_result(data_processed_train_dict,data_processed_validate_dict)
