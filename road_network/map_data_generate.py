import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import networkx as nx
import math
from collections import Counter
from pre_processing.road_network import load_rn_shp
from pre_processing.map import get_candi_shl


road_data_path = '/home/jnli/SHL_2023/SHL2023/road_network/data'
edge_g = nx.read_gpickle('{}/graph_shl.gpickle'.format(road_data_path))
roadid2code = np.load('{}/roadid2code.npy'.format(road_data_path),allow_pickle=True).item()
# 生成需要的地图
rn = load_rn_shp(edge_g,is_directed=True)# 加载路网数据





# print('-----------------------1 get validataion  Bag---------------------')
# data_path = '/home/jnli/SHL_2023/SHL2023/data2023/validate/Bag'
# feature_data_path = '{}/data_processed_validate_1.csv'.format(data_path)
# save_path = data_path# 数据路径
# df_point = pd.read_csv(feature_data_path)
# get_candi_shl(df_point,rn,roadid2code,save_path,search_dis=50)
# print('map data get well')

# print('-----------------------1 get validataion  Hips---------------------')
# data_path = '/home/jnli/SHL_2023/SHL2023/data2023/validate/Hips'
# feature_data_path = '{}/data_processed_validate_1.csv'.format(data_path)
# save_path = data_path# 数据路径
# df_point = pd.read_csv(feature_data_path)
# get_candi_shl(df_point,rn,roadid2code,save_path,search_dis=50)
# print('map data get well')

# print('-----------------------1 get validataion  Torso---------------------')
# data_path = '/home/jnli/SHL_2023/SHL2023/data2023/validate/Torso'
# feature_data_path = '{}/data_processed_validate_1.csv'.format(data_path)
# save_path = data_path# 数据路径
# df_point = pd.read_csv(feature_data_path)
# get_candi_shl(df_point,rn,roadid2code,save_path,search_dis=50)
# print('map data get well')

# print('-----------------------1 get validataion  Hand---------------------')
# data_path = '/home/jnli/SHL_2023/SHL2023/data2023/validate/Hand'
# feature_data_path = '{}/data_processed_validate_1.csv'.format(data_path)
# save_path = data_path# 数据路径
# df_point = pd.read_csv(feature_data_path)
# get_candi_shl(df_point,rn,roadid2code,save_path,search_dis=50)
# print('map data get well')





# print('-----------------------2  get train  Bag---------------------')
# data_path = '/home/jnli/SHL_2023/SHL2023/data2023/train/Bag'
# feature_data_path = '{}/data_processed_train_1.csv'.format(data_path)
# save_path = data_path# 数据路径
# df_point = pd.read_csv(feature_data_path)
# get_candi_shl(df_point,rn,roadid2code,save_path,search_dis=50)
# print('map data get well')

# print('-----------------------2  get train  Hand---------------------')
# data_path = '/home/jnli/SHL_2023/SHL2023/data2023/train/Hand'
# feature_data_path = '{}/data_processed_train_1.csv'.format(data_path)
# save_path = data_path# 数据路径
# df_point = pd.read_csv(feature_data_path)
# get_candi_shl(df_point,rn,roadid2code,save_path,search_dis=50)
# print('map data get well')

# print('-----------------------2  get train  Hips---------------------')
# data_path = '/home/jnli/SHL_2023/SHL2023/data2023/train/Hips'
# feature_data_path = '{}/data_processed_train_1.csv'.format(data_path)
# save_path = data_path# 数据路径
# df_point = pd.read_csv(feature_data_path)
# get_candi_shl(df_point,rn,roadid2code,save_path,search_dis=50)
# print('map data get well')
# print('-----------------------2  get train  Torso---------------------')
# data_path = '/home/jnli/SHL_2023/SHL2023/data2023/train/Torso'
# feature_data_path = '{}/data_processed_train_1.csv'.format(data_path)
# save_path = data_path# 数据路径
# df_point = pd.read_csv(feature_data_path)
# get_candi_shl(df_point,rn,roadid2code,save_path,search_dis=50)
# print('map data get well')




print('-----------------------3  get test  ---------------------')
data_path = '/home/jnli/SHL_2023/SHL2023/data2023/test'
feature_data_path = '{}/data_processed_test_1.csv'.format(data_path)
save_path = data_path# 数据路径
df_point = pd.read_csv(feature_data_path)
get_candi_shl(df_point,rn,roadid2code,save_path,search_dis=50)
print('map data get well')