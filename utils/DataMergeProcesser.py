import pandas as pd 
import numpy as np
from collections import Counter
from utils.DataLoader import SHLDataLoader
import utils.helper as helper
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.signal import welch
from utils.prepare import fill_na, calculate_abs_values, calculate_change, calculate_pct_change, calculate_window, calculate_shift
fill_limit = 100


settings = {
    'fill_limit': 100,
    'window_sizes': [20],
    'window_center': True,
    'columns': ['ACC_x_mean',
 'ACC_x_max',
 'ACC_x_min',
 'ACC_x_std',
 'ACC_x_median',
 'ACC_x_cal_range',
 'ACC_y_mean',
 'ACC_y_max',
 'ACC_y_min',
 'ACC_y_std',
 'ACC_y_median',
 'ACC_y_cal_range',
 'ACC_z_mean',
 'ACC_z_max',
 'ACC_z_min',
 'ACC_z_std',
 'ACC_z_median',
 'ACC_z_cal_range',
 'GYR_x_mean',
 'GYR_x_max',
 'GYR_x_min',
 'GYR_x_std',
 'GYR_x_median',
 'GYR_x_cal_range',
 'GYR_y_mean',
 'GYR_y_max',
 'GYR_y_min',
 'GYR_y_std',
 'GYR_y_median',
 'GYR_y_cal_range',
 'GYR_z_mean',
 'GYR_z_max',
 'GYR_z_min',
 'GYR_z_std',
 'GYR_z_median',
 'GYR_z_cal_range',
 'MAG_x_mean',
 'MAG_x_max',
 'MAG_x_min',
 'MAG_x_std',
 'MAG_x_median',
 'MAG_x_cal_range',
 'MAG_y_mean',
 'MAG_y_max',
 'MAG_y_min',
 'MAG_y_std',
 'MAG_y_median',
 'MAG_y_cal_range',
 'MAG_z_mean',
 'MAG_z_max',
 'MAG_z_min',
 'MAG_z_std',
 'MAG_z_median',
 'MAG_z_cal_range',
 'norm_mag_mean',
 'norm_mag_max',
 'norm_mag_min',
 'norm_mag_std',
 'norm_mag_median',
 'norm_mag_cal_range',
 'norm_acc_mean',
 'norm_acc_max',
 'norm_acc_min',
 'norm_acc_std',
 'norm_acc_median',
 'norm_acc_cal_range',
 'norm_gyr_mean',
 'norm_gyr_max',
 'norm_gyr_min',
 'norm_gyr_std',
 'norm_gyr_median',
 'norm_gyr_cal_range',
 'ACC_x_avg_frequency',
 'ACC_x_peak_frequency',
 'ACC_x_kurtosis',
 'ACC_x_skewness',
 'ACC_x_energy',
 'ACC_x_entropy',
 'ACC_y_avg_frequency',
 'ACC_y_peak_frequency',
 'ACC_y_kurtosis',
 'ACC_y_skewness',
 'ACC_y_energy',
 'ACC_y_entropy',
 'ACC_z_avg_frequency',
 'ACC_z_peak_frequency',
 'ACC_z_kurtosis',
 'ACC_z_skewness',
 'ACC_z_energy',
 'ACC_z_entropy',
 'GYR_x_avg_frequency',
 'GYR_x_peak_frequency',
 'GYR_x_kurtosis',
 'GYR_x_skewness',
 'GYR_x_energy',
 'GYR_x_entropy',
 'GYR_y_avg_frequency',
 'GYR_y_peak_frequency',
 'GYR_y_kurtosis',
 'GYR_y_skewness',
 'GYR_y_energy',
 'GYR_y_entropy',
 'GYR_z_avg_frequency',
 'GYR_z_peak_frequency',
 'GYR_z_kurtosis',
 'GYR_z_skewness',
 'GYR_z_energy',
 'GYR_z_entropy',
 'MAG_x_avg_frequency',
 'MAG_x_peak_frequency',
 'MAG_x_kurtosis',
 'MAG_x_skewness',
 'MAG_x_energy',
 'MAG_x_entropy',
 'MAG_y_avg_frequency',
 'MAG_y_peak_frequency',
 'MAG_y_kurtosis',
 'MAG_y_skewness',
 'MAG_y_energy',
 'MAG_y_entropy',
 'MAG_z_avg_frequency',
 'MAG_z_peak_frequency',
 'MAG_z_kurtosis',
 'MAG_z_skewness',
 'MAG_z_energy',
 'MAG_z_entropy',
 'norm_acc_avg_frequency',
 'norm_acc_peak_frequency',
 'norm_acc_kurtosis',
 'norm_acc_skewness',
 'norm_acc_energy',
 'norm_acc_entropy',
 'norm_gyr_avg_frequency',
 'norm_gyr_peak_frequency',
 'norm_gyr_kurtosis',
 'norm_gyr_skewness',
 'norm_gyr_energy',
 'norm_gyr_entropy',
 'norm_mag_avg_frequency',
 'norm_mag_peak_frequency',
 'norm_mag_kurtosis',
 'norm_mag_skewness',
 'norm_mag_energy',
 'norm_mag_entropy',
 'number',
 'accuracy',
 'latitude',
 'longitude',
 'altitude',
 'gps_snr_mean',
 'gps_snr_min',
 'gps_snr_max',
 'gps_snr_std'], #['distance_London', 'distance_Brighton', 'distance_parks', 'bus_stops'],
    'functions': ['mean', 'std'],
}
# 上面的窗口特征最后只留了这么多，其他都效果不好
# window_columns = ["norm_gyr_median_window_20_mean",
# "norm_acc_std_window_20_mean",
# "gps_snr_mean_window_20_mean",
# "norm_mag_peak_frequency_window_20_mean",
# "number_window_20_mean",
# "norm_gyr_min_window_20_mean",
# "ACC_y_std_window_20_mean",
# "GYR_y_std_window_20_mean",
# "longitude_window_20_mean",
# "longitude_window_20_std"]
def cal_range(column):
    return column.max() - column.min()

class DataMergeProcesser():
    def __init__(self, data_path):
        # super(self.DataProcesser).__init__()
        
        self.data_path = data_path
        self.df_label = pd.read_csv('{}/get_road_feature_1.csv'.format(data_path))
        print('data load well')

        self.df_label.rename(columns={'speed': 'speed_square'}, inplace=True)
        self.df =  self.df_label
        print('raw length: ',len(self.df))
        print('------------------------1 Data preparation completed-----------------------')
        self.colum_list = list(self.df.columns)
    def get_more_loc_fea(self):
        old_features = set(list(self.df))
        # prepare
        self.df['time_dlt'] = self.df['time'].diff().fillna(method = 'bfill')
        # Additional features
        df_diff = calculate_change(self.df[['latitude', 'longitude', 'altitude']], 100)
        df_diff['distance'] = (df_diff['latitude_change'].pow(2) + df_diff['longitude_change'].pow(2)).pow(0.5)
        df_diff['speed'] = (df_diff['distance']/self.df['time_dlt'])
        df_diff['is_stop'] = df_diff['speed'].apply(lambda x: 1 if x == 0 else 0)
        df_diff.drop(['latitude_change', 'longitude_change', 'altitude_change'], axis=1, inplace=True) 
        fill_na(df_diff, 100)
        self.df =  pd.concat([self.df,df_diff], axis=1)
        self.df= self.df.fillna(method='ffill').fillna(method='bfill')
        new_features = set(list(self.df)).difference(old_features)
        print("------------------------ 2  generate more loc features ------------------------")
        new_features = set(list(self.df)).difference(old_features)
        print("New Feature Added: {}".format(new_features))
        print('after get more loc length: ',len(self.df))
        self.colum_list = list(self.df.columns)
        print('more loc features saved well')

    def get_change_windows_features(self):

        old_features = set(list(self.df))

        print("------------------------2  get changed  features  ------------------------")
        df_diff = calculate_change(self.df[['speed','distance', 'distance_London','distance_Brighton','distance_parks','distance_bus_stops','distance_bus_routes','distance_subway','distance_railways','error']], fill_limit)          
        df_pct = calculate_pct_change(self.df[['accuracy']])
        self.df = pd.concat([self.df,df_diff, df_pct], axis=1)
        self.df= self.df.fillna(method='ffill').fillna(method='bfill')
        
        self.df.to_csv('{}/get_experiment_features.csv'.format(self.data_path))

        print("------------------------3  get  windows features  ------------------------")
        self.df = calculate_window(self.df, **settings)  
        self.df= self.df.fillna(method='ffill').fillna(method='bfill')

        new_features = set(list(self.df)).difference(old_features)
        print("------------------------ changed and windows features get well ------------------------")
        print("New Feature Added: {}".format(new_features))
        print('after change and windows features length: ',len(self.df))
        self.df.to_csv('{}/get_experiment_features_1.csv'.format(self.data_path))
        print('last all  features saved well')



