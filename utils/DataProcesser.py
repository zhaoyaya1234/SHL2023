import pandas as pd 
import numpy as np
from collections import Counter
from utils.DataLoader import SHLDataLoader
import utils.helper as helper
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.signal import welch

def cal_range(column):
    return column.max() - column.min()

class DataProcesser():
    def __init__(self, dataloader):
        # super(self.DataProcesser).__init__()
        self.data = dataloader
        if isinstance(self.data, SHLDataLoader):
            print("Customized DataLoader Passed in.")
        else:
            print("Wrong data type {}. Abort!".format(type(self.data)))

    def process_pre(self):
        # unit: 1s
    
        
        self.data.loc['time'] = self.data.loc['time'].astype('int').apply(pd.to_numeric)
        self.data.gps['time'] = self.data.gps['time'].astype('int').apply(pd.to_numeric)

        self.data.gyr['time'] = self.data.gyr['time'].astype('int').apply(pd.to_numeric)
        self.data.mag['time'] = self.data.mag['time'].astype('int').apply(pd.to_numeric)
        self.data.acc['time'] = self.data.acc['time'].astype('int').apply(pd.to_numeric)
        

        try:
            print("Labeled Dataset Detected")
            
            self.data.label['time'] = self.data.label['time'].astype('int').apply(pd.to_numeric)

            assert self.data.gyr['time'].equals(self.data.mag['time']) and self.data.mag['time'].equals(self.data.acc['time']) and self.data.mag['time'].equals(self.data.label['time'])

            self.data.df_label = pd.concat([self.data.acc, self.data.gyr.drop('time', axis=1)], axis=1)
            self.data.df_label = pd.concat([self.data.df_label, self.data.mag.drop('time', axis=1)], axis=1)
            self.data.df_label = pd.concat([self.data.df_label, self.data.label.drop('time', axis=1)], axis=1)

        except:
            print("Unlabeled Dataset Detected")

            assert self.data.gyr['time'].equals(self.data.mag['time']) and self.data.mag['time'].equals(self.data.acc['time']) 
            
            self.data.df_label = pd.concat([self.data.acc, self.data.gyr.drop('time', axis=1)], axis=1)
            self.data.df_label = pd.concat([self.data.df_label, self.data.mag.drop('time', axis=1)], axis=1)
        
        print("------------------------ 1  Basic Features Extracted (data.df_label) ------------------------")
        print("df_label Feature Initialized: {}".format(list(self.data.df_label)))
        print('simple cancat length',len(self.data.df_label))
    def generate_new_fea(self):

        old_features = set(list(self.data.df_label))
        self.data.df_label = self.data.df_label.interpolate()
        self.data.df_label['norm_mag'] = self.data.df_label.apply(lambda row: row['MAG_x']**2 + row['MAG_y']**2 + row['MAG_z']**2, axis=1)
        self.data.df_label['norm_acc'] = self.data.df_label.apply(lambda row: row['ACC_x']**2 + row['ACC_y']**2 + row['ACC_z']**2, axis=1)
        self.data.df_label['norm_gyr'] = self.data.df_label.apply(lambda row: row['GYR_x']**2 + row['GYR_y']**2 + row['GYR_z']**2, axis=1)

        new_features = set(list(self.data.df_label)).difference(old_features)
        print("------------------------ 2 generate features from mag gyr acc ------------------------")
        print("New Feature Added: {}".format(new_features))
    

    def process_df_label(self):
        print('raw length: ',len(self.data.df_label))
        try:
            print("Labeled Dataset Detected")
            df_numeric = self.data.df_label.drop(['time', 'label'], axis=1)

            # 计算每一百行的统计指标并添加标签列和时间戳均值列
            n = 100
            df_segmented = df_numeric.groupby(df_numeric.index // n).agg(['mean', 'max', 'min', 'std', 'median',cal_range])
            
            for column in ['ACC_x', 'ACC_y', 'ACC_z', 'GYR_x', 'GYR_y', 'GYR_z', 'MAG_x', 'MAG_y', 'MAG_z','norm_acc', 'norm_gyr', 'norm_mag']:
      
                data_windows = [df_numeric[column].iloc[i * n: (i + 1) * n] for i in range(len(df_segmented))]
                freqs, powers = zip(*[welch(window) for window in data_windows])
                
                # 计算频率和功率的统计指标
                avg_frequencies = np.array([np.sum(freq * power) / np.sum(power) for freq, power in zip(freqs, powers)])
                peak_frequencies = np.array([freq[np.argmax(power)] for freq, power in zip(freqs, powers)])
                kurtosis_values = np.array([window.kurtosis() for window in data_windows])
                skewness_values = np.array([window.skew() for window in data_windows])
                energy_values = np.array([np.sum(power) for power in powers])
                
                # 计算修正后的熵
                min_nonzero_power = np.min(np.concatenate([power[power > 0] for power in powers]))  # 获取最小非零功率值
                powers = [np.where(power <= 0, min_nonzero_power, power) for power in powers]
                entropy_values = np.array([-np.sum(power * np.log2(power)) for power in powers])
        
                
                # 添加频率特征的列
                df_segmented[(column, 'avg_frequency')] = avg_frequencies
                df_segmented[(column, 'peak_frequency')] = peak_frequencies
                df_segmented[(column, 'kurtosis')] = kurtosis_values
                df_segmented[(column, 'skewness')] = skewness_values
                df_segmented[(column, 'energy')] = energy_values
                df_segmented[(column, 'entropy')] = entropy_values
            
            
            df_segmented.columns = df_segmented.columns.map('_'.join)

            # 计算每个标签的众数
            label_modes = [self.data.df_label['label'].iloc[i * n: (i + 1) * n].value_counts().idxmax() for i in range(len(df_segmented))]
            # label_modes_count = [df['label'].iloc[i * n: (i + 1) * n].value_counts() for i in range(len(df_segmented))]
            # 添加标签列、时间戳均值列和标签众数列
            df_segmented['label_all'] = [self.data.df_label['label'].iloc[i * n: (i + 1) * n].tolist() for i in range(len(df_segmented))]
            df_segmented['time_all'] = [self.data.df_label['time'].iloc[i * n: (i + 1) * n].tolist() for i in range(len(df_segmented))]
            df_segmented['time_mean'] = [self.data.df_label['time'].iloc[i * n: (i + 1) * n].mean() for i in range(len(df_segmented))]
            df_segmented['time_mean'] = df_segmented['time_mean'].astype('int').apply(pd.to_numeric)
            df_segmented['label_mode'] = label_modes

            last_segment_rows = len(self.data.df_label) % n
            df_segmented.loc[df_segmented.index[-1], 'label_num'] = last_segment_rows if last_segment_rows != 0 else n
            df_segmented['label_num'] = df_segmented['label_num'].fillna(n).astype(int)

            
            self.data.df_label = df_segmented
            self.data.df_label.rename(columns={'time_mean': 'time'}, inplace=True)

        except:
            print("Unlabeled Dataset Detected")

            df_numeric = self.data.df_label.drop(['time'], axis=1)

            # 计算每一百行的统计指标并添加标签列和时间戳均值列
            n = 100
            df_segmented = df_numeric.groupby(df_numeric.index // n).agg(['mean', 'max', 'min', 'std', 'median',cal_range])
            
            for column in ['ACC_x', 'ACC_y', 'ACC_z', 'GYR_x', 'GYR_y', 'GYR_z', 'MAG_x', 'MAG_y', 'MAG_z','norm_acc', 'norm_gyr', 'norm_mag']:
      
                data_windows = [df_numeric[column].iloc[i * n: (i + 1) * n] for i in range(len(df_segmented))]
                freqs, powers = zip(*[welch(window) for window in data_windows])
                
                # 计算频率和功率的统计指标
                avg_frequencies = np.array([np.sum(freq * power) / np.sum(power) for freq, power in zip(freqs, powers)])
                peak_frequencies = np.array([freq[np.argmax(power)] for freq, power in zip(freqs, powers)])
                
                kurtosis_values = np.array([window.kurtosis() for window in data_windows])
                skewness_values = np.array([window.skew() for window in data_windows])
                energy_values = np.array([np.sum(power) for power in powers])
                
                # 计算修正后的熵
                min_nonzero_power = np.min(np.concatenate([power[power > 0] for power in powers]))  # 获取最小非零功率值
                powers = [np.where(power <= 0, min_nonzero_power, power) for power in powers]
                entropy_values = np.array([-np.sum(power * np.log2(power)) for power in powers])

                
                # 添加频率特征的列
                df_segmented[(column, 'avg_frequency')] = avg_frequencies
                df_segmented[(column, 'peak_frequency')] = peak_frequencies
                df_segmented[(column, 'kurtosis')] = kurtosis_values
                df_segmented[(column, 'skewness')] = skewness_values
                df_segmented[(column, 'energy')] = energy_values
                df_segmented[(column, 'entropy')] = entropy_values
            
            df_segmented.columns = df_segmented.columns.map('_'.join)

           
            # 添加标签列、时间戳均值列和标签众数列
            df_segmented['time_all'] = [self.data.df_label['time'].iloc[i * n: (i + 1) * n].tolist() for i in range(len(df_segmented))]
            df_segmented['time_mean'] = [self.data.df_label['time'].iloc[i * n: (i + 1) * n].mean() for i in range(len(df_segmented))]
            df_segmented['time_mean'] = df_segmented['time_mean'].astype('int').apply(pd.to_numeric)

            last_segment_rows = len(self.data.df_label) % n
            df_segmented.loc[df_segmented.index[-1], 'label_num'] = last_segment_rows if last_segment_rows != 0 else n
            df_segmented['label_num'] = df_segmented['label_num'].fillna(n).astype(int)
            self.data.df_label = df_segmented
            self.data.df_label.rename(columns={'time_mean': 'time'}, inplace=True)
        new_features = set(list(self.data.df_label))
        print("------------------------3  ACC MAG  GYR features/100 Extracted ------------------------")
        print("New Feature Added: {}".format(new_features))
        print('after /100 length: ',len(self.data.df_label))
    
    def merge_data(self):

        old_features = set(list(self.data.df_label))
        

        self.data.gps['number'] = self.data.gps['number'].astype('int').apply(pd.to_numeric)
        df1 = self.data.df_label
        df2 = self.data.gps
        df3 = self.data.loc

        df1['time_series1'] = pd.to_datetime(df1['time'], unit='ms').dt.floor('S')
        df2['time_series2'] = pd.to_datetime(df2['time'], unit='ms').dt.floor('S')
        df3['time_series3'] = pd.to_datetime(df3['time'], unit='ms').dt.floor('S')
    
        df2 = df2.groupby(['time_series2'], as_index = False).mean()
        df3 = df3.groupby(['time_series3'], as_index = False).mean()

        # 按照时间戳合并DataFrame
        merged_df = pd.merge(df1, df2, left_on=df1['time_series1'], right_on=df2['time_series2'], how='left')
        merged_df = merged_df.drop('key_0', axis=1)
       
        merged_df = pd.merge(merged_df, df3.rename({'time':'time_z'}, axis = 1), left_on=merged_df['time_series1'], right_on=df3['time_series3'], how='left')
        # # 删除重复的时间戳列
        merged_df = merged_df.drop('key_0', axis=1)
       
        merged_df.rename(columns={'time_x': 'time'}, inplace=True)
        self.data.df = merged_df
        self.data.df['availability'] = np.where(self.data.df.isna().any(axis=1), 0, 1)
        
        new_features = set(list(self.data.df)).difference(old_features)
        print("------------------------4  gps_loc avali sensors Features merged  ------------------------")
        print("New Feature Added: {}".format(new_features))
        print('after merge length: ',len(self.data.df))
    def process_gps(self):
        old_features = set(list(self.data.df))
        self.data.gps_detail['time'] = self.data.gps_detail['time'].astype('int').apply(pd.to_numeric)
        self.data.gps_detail['time_series2'] = pd.to_datetime(self.data.gps_detail['time'], unit='ms').dt.floor('S')

        self.data.gps_detail['gps_snr'] = self.data.gps_detail['snr'].apply(pd.to_numeric)

        tmp_gps_mean = self.data.gps_detail[['time_series2', 'gps_snr']].groupby(['time_series2'], as_index = False).mean().add_suffix("_mean")
        tmp_gps_min = self.data.gps_detail[['time_series2', 'gps_snr']].groupby(['time_series2'], as_index = False).min().add_suffix("_min")
        tmp_gps_max = self.data.gps_detail[['time_series2', 'gps_snr']].groupby(['time_series2'], as_index = False).max().add_suffix("_max")
        tmp_gps_std = self.data.gps_detail[['time_series2', 'gps_snr']].groupby(['time_series2'], as_index = False).std().add_suffix("_std")

        tmp_gps = pd.merge(tmp_gps_mean.rename({"time_series2_mean": "time_series2"}, axis = 1), tmp_gps_min.rename({"time_series2_min": "time_series2"}, axis = 1), on = ['time_series2'])
        tmp_gps = pd.merge(tmp_gps, tmp_gps_max.rename({"time_series2_max": "time_series2"}, axis = 1), on = ['time_series2'])
        tmp_gps = pd.merge(tmp_gps, tmp_gps_std.rename({"time_series2_std": "time_series2"}, axis = 1), on = ['time_series2'])
        
        self.data.df = pd.merge(self.data.df, tmp_gps, on = ['time_series2'], how = 'left')
        new_features = set(list(self.data.df)).difference(old_features)
        print("------------------------5  Features in self.data.gps(_detail) Extracted ------------------------")
        print("New Feature Added: {}".format(new_features))
        print('after gps length: ',len(self.data.df))

    def process_loc(self):
        old_features = set(list(self.data.df))
        # prepare
        self.data.df['time_dlt'] = self.data.df['time'].diff().fillna(method = 'bfill')

        self.data.df['valid_dlt'] = self.data.df.apply(lambda x: int(x['time_dlt'] <= 10000 and x['time_dlt'] >= 1), axis = 1)
        # utm loc
        self.data.df['east'] = self.data.df.apply(lambda x: helper.gps2utm_east(x), axis = 1)
        self.data.df['north'] = self.data.df.apply(lambda x: helper.gps2utm_north(x), axis = 1)
        self.data.df['east_dlt'] = self.data.df['east'].diff(1)
        self.data.df['north_dlt'] = self.data.df['north'].diff(1)
        # speed
        self.data.df['east_speed'] = self.data.df.apply(lambda x: x['east_dlt']/(x['time_dlt']/1000) if x['valid_dlt'] == 1 else np.nan, axis = 1)
        self.data.df['north_speed'] = self.data.df.apply(lambda x: x['north_dlt']/(x['time_dlt']/1000) if x['valid_dlt'] == 1 else np.nan, axis = 1)
        self.data.df['east_speed'] = self.data.df['east_speed'].apply(lambda x: x if np.abs(x) < 300 else np.nan)
        self.data.df['north_speed'] = self.data.df['north_speed'].apply(lambda x: x if np.abs(x) < 300 else np.nan)
        self.data.df['speed'] = self.data.df.apply(lambda x: np.sqrt(x['east_speed']**2 + x['north_speed']**2), axis = 1)
        self.data.df['speed_dif'] = self.data.df.apply(lambda x: np.abs(x['east_speed'] - x['north_speed']), axis = 1)
        # acc 
        self.data.df['speed_dlt'] = self.data.df['speed'].diff(1)
        self.data.df['acc'] = self.data.df.apply(lambda x: x['speed_dlt']/(x['time_dlt']/1000) if x['valid_dlt'] == 1  else np.nan, axis = 1)
        
        new_features = set(list(self.data.df)).difference(old_features)
        print("------------------------ 6 Features in self.data.loc Extracted ------------------------")
        print("New Feature Added: {}".format(new_features))
        print('after loc length: ',len(self.data.df))

    
    def process_data_more(self):
        old_features = set(list(self.data.df))
        self.data.df['speed_log'] = self.data.df.apply(lambda x: np.log(x['speed'] + 1), axis = 1) 
        self.data.df['speed_wd_std'] = self.data.df[['speed']].rolling(10).std()['speed']
        self.data.df['speed_wd_max_log'] = np.log(self.data.df['speed']+1).rolling(20, center = True).max().fillna(method = 'ffill').fillna(method = 'bfill')
        
        
        self.data.df['acc_wd_std'] = self.data.df[['acc']].rolling(10).std()['acc']
        self.data.df['acc_wd_max_log'] = np.log(self.data.df['acc']+1).rolling(20, center = True).max().fillna(method = 'ffill').fillna(method = 'bfill')
        
        self.data.df = self.data.df.fillna(method='ffill').fillna(method='bfill')
       
        
        
        new_features = set(list(self.data.df)).difference(old_features)
        print("------------------------ 7  More New Features Extracted form gps and loc ------------------------")
        print("New Feature Added: {}".format(new_features))
        print('after get more length: ',len(self.data.df))

    
    def process_pipe(self):
    
        self.process_pre()
        self.generate_new_fea()
        self.process_df_label()

        self.merge_data()
    
        self.process_gps()
        self.process_loc()
        self.process_data_more()