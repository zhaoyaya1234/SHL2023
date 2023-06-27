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
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
from utils.DataLoader import SHLDataLoader
# from utils.DataProcesser import DataProcesser
from utils.DataMergeProcesser import DataMergeProcesser





print('------------------------1  validataion   Hand  (1/4) --------------------------')
data_path  = '/home/jnli/SHL_2023/SHL2023/data2023/validate/Hand'
print(os.listdir(data_path))
merge_data = DataMergeProcesser(data_path)
merge_data.get_more_loc_fea()
merge_data.get_change_windows_features()

print('------------------------1  validataion   Bag (2/4) --------------------------')
data_path  = '/home/jnli/SHL_2023/SHL2023/data2023/validate/Bag'
print(os.listdir(data_path))
merge_data = DataMergeProcesser(data_path)
merge_data.get_more_loc_fea()
merge_data.get_change_windows_features()
# 验证集3 
print('------------------------1  validataion  Hips  (3/4) --------------------------')
data_path  = '/home/jnli/SHL_2023/SHL2023/data2023/validate/Hips'
print(os.listdir(data_path))
merge_data = DataMergeProcesser(data_path)
merge_data.get_more_loc_fea()
merge_data.get_change_windows_features()
# 验证集4
print('------------------------1  validataion  Torso (4/4) -----------------------------')
data_path  = '/home/jnli/SHL_2023/SHL2023/data2023/validate/Torso'
print(os.listdir(data_path))
merge_data = DataMergeProcesser(data_path)
merge_data.get_more_loc_fea()
merge_data.get_change_windows_features()


# 训练集2
print('-------------------------3  train Bag  (1/4) --------------------------------------')
data_path =  '/home/jnli/SHL_2023/SHL2023/data2023/train/Bag'
print(os.listdir(data_path))
merge_data = DataMergeProcesser(data_path)
merge_data.get_more_loc_fea()
merge_data.get_change_windows_features()

print('-------------------------3  train   Hand   (2/4) --------------------------------------')
data_path =  '/home/jnli/SHL_2023/SHL2023/data2023/train/Hand'
print(os.listdir(data_path))
merge_data = DataMergeProcesser(data_path)
merge_data.get_more_loc_fea()
merge_data.get_change_windows_features()

# 训练集3
print('-------------------------3  train   Hips   (3/4)--------------------------------------')
data_path =  '/home/jnli/SHL_2023/SHL2023/data2023/train/Hips'
print(os.listdir(data_path))
merge_data = DataMergeProcesser(data_path)
merge_data.get_more_loc_fea()
merge_data.get_change_windows_features()

# 训练集4
print('-------------------------3  train Torso   (4/4)--------------------------------------')
data_path =  '/home/jnli/SHL_2023/SHL2023/data2023/train/Torso'
print(os.listdir(data_path))
merge_data = DataMergeProcesser(data_path)
merge_data.get_more_loc_fea()
merge_data.get_change_windows_features()

print('-------------------------2 test--------------------------------------')
data_path = '/home/jnli/SHL_2023/SHL2023/data2023/test'
print(os.listdir(data_path))
merge_data = DataMergeProcesser(data_path)
merge_data.get_more_loc_fea()
merge_data.get_change_windows_features()