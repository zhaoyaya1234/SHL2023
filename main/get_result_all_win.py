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

drop_columns =  ['Unnamed: 0',
 'label_mode',
 ]

label_dict = {1: 'Still', 2: 'Walking', 3: 'Run', 4: 'Bike', 5: 'Car', 6: 'Bus', 7: 'Train', 8: 'Subway'}

# # 二 读取特征信息

def get_valid_pre_result(data_processed_train_dict,data_processed_validate_dict):
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
            print('training set : {} ; testing set : {}  testing result'.format(train_type,test_type))
            for window in [80,120,150,180,200,300]:
                tree_model.predict_mean(window)
                print('tree_model.pred_mean with window : {}'.format(window))
                evaluate(y_test, tree_model.pred_mean, normalize = True, names = list(label_dict.values()))


data_path_train_1 =  '/home/jnli/SHL_2023/SHL2023/data2023/train/Bag'
data_path_train_2 =  '/home/jnli/SHL_2023/SHL2023/data2023/train/Hand'
data_path_train_3 =  '/home/jnli/SHL_2023/SHL2023/data2023/train/Hips'
data_path_train_4 =  '/home/jnli/SHL_2023/SHL2023/data2023/train/Torso'

data_path_valid_1 = '/home/jnli/SHL_2023/SHL2023/data2023/validate/Bag'
data_path_valid_2 = '/home/jnli/SHL_2023/SHL2023/data2023/validate/Hand'
data_path_valid_3 = '/home/jnli/SHL_2023/SHL2023/data2023/validate/Hips'
data_path_valid_4 = '/home/jnli/SHL_2023/SHL2023/data2023/validate/Torso'

data_processed_train_1 = pd.read_csv('{}/get_last_features_banlenced.csv'.format(data_path_train_1))
data_processed_train_2 = pd.read_csv('{}/get_last_features_banlenced.csv'.format(data_path_train_2))
data_processed_train_3 = pd.read_csv('{}/get_last_features_banlenced.csv'.format(data_path_train_3))
data_processed_train_4 = pd.read_csv('{}/get_last_features_banlenced.csv'.format(data_path_train_4))
data_processed_train = pd.concat([data_processed_train_1, data_processed_train_2, data_processed_train_3, data_processed_train_4], axis=0)

print('the all length of training set:{}'.format(len(data_processed_train)))

data_processed_validate_1 = pd.read_csv('{}/get_last_features_banlenced.csv'.format(data_path_valid_1))
data_processed_validate_2 = pd.read_csv('{}/get_last_features_banlenced.csv'.format(data_path_valid_2))
data_processed_validate_3 = pd.read_csv('{}/get_last_features_banlenced.csv'.format(data_path_valid_3))
data_processed_validate_4 = pd.read_csv('{}/get_last_features_banlenced.csv'.format(data_path_valid_4))
data_processed_validate  = pd.concat([data_processed_validate_1, data_processed_validate_2, data_processed_validate_3, data_processed_validate_4], axis=0)
print('the all length of test set:{}'.format(len(data_processed_validate)))

print('begin training single')

data_processed_train_dict = {'Hips':data_processed_train_3,'Bag':data_processed_train_1,'Hand':data_processed_train_2,'Torso':data_processed_train_4}
data_processed_validate_dict = {'Torso':data_processed_validate_4,'Bag':data_processed_validate_1,'Hand':data_processed_validate_2,'Hips':data_processed_validate_3}
get_valid_pre_result(data_processed_train_dict,data_processed_validate_dict)

print('begin training all')

data_processed_train_dict = {'all_train_set':data_processed_train}
data_processed_validate_dict = {'all_test_set':data_processed_validate,'Torso':data_processed_validate_4,'Bag':data_processed_validate_1,'Hand':data_processed_validate_2,'Hips':data_processed_validate_3}
get_valid_pre_result(data_processed_train_dict,data_processed_validate_dict)
