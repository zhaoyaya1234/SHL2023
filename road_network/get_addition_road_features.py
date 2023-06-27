import re
import numpy as np
import pandas as pd
import os
from shl.prepare import normalize_epoch_time, normalize_lat_long, calculate_window, calculate_shift, fillna_agg_by_label
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import haversine_distances
from sklearn.metrics import pairwise_distances_argmin_min
from shapely.geometry import Polygon, LineString, Point, MultiLineString
from tqdm import tqdm
import geopandas as gpd
import pickle
import branca.colormap as cm


# 1 点特征
with open('./additional_data/bus_stops.pickle', 'rb') as f:
    bus_stops = pickle.load(f)

feature_points = {
    'London': [
        [51.509865, -0.118092]
    ],
    'Brighton': [
        [50.827778, -0.152778]
    ],
    'parks': [
        # Hollingbury Golf Course
        [50.85423803467499, -0.12791258170001926],
        # St Ann’s Well Gardens
        [50.829876823789675, -0.15525600010959892],
        # Preston Park
        [50.839694335541274, -0.1462660790420134],
        # Waterwall conservation area
        [50.8659, -0.1750],
        # Withdean park
        [50.8546, -0.1497],
        # Stanmer park
        [50.8678, -0.0968],
        # Upper Lodge Wood
        [50.8751, -0.1177],
        # Pudding bag
        [50.8710, -0.1161],
        # Great Wood
        [50.8653, -0.1036],
        # Grubbings
        [50.8729, -0.0971],
        # Milbark wood
        [50.8783, -0.0982],
        # High park wood
        [50.8849, -0.1078],
        # Green broom
        [50.8833, -0.1107],
        # Moon's plantations
        [50.8774, -0.0840],
        # Friston forest
        [50.7783, 0.1894],
        # Malthouse wood
        [51.0042, -0.2044],
        # Bedgebury forest
        [51.0694, 0.4649],
        # Swinley forest
        [51.3726, -0.7292],
        # Crowthore wood
        [51.3808, -0.7598],
        # Queen Elizabeth Country Parh
        [50.9651, -0.9695],
        # Hurth wood
        [51.1854, -0.4278],
        # Winterfold wood
        [51.1764, -0.4564],
        # Queen's park
        [50.8249, -0.1248],
    ],
    'bus_stops': bus_stops
}

def calculate_minimal_distance(data: pd.DataFrame, points):
    from sklearn.neighbors import BallTree
    tree = BallTree(np.array(points), leaf_size=15)
    distances, indices = tree.query(data[['latitude','longitude']], k=1)
    return distances
    # return pairwise_distances_argmin_min(data[['Latitude','Longitude']], np.array(points))

def create_point_distance_features(data: pd.DataFrame):
    # features = data[['epoch_time_id']]
    features = data
    for name, points in feature_points.items():
        if len(points) > 0:
            if type(points[0]) == list: 
                features[f'distance_{name}'] = calculate_minimal_distance(data, points)
    return features


# 2 获得离地铁，火车的最近距离
railways = gpd.read_file('./additional_data/railways.json')
# display(railways)
subway = gpd.read_file('./additional_data/subway.json')
# display(subway)
from shapely.ops import unary_union, cascaded_union, linemerge

feature_lines = {
    # 'bus_routes': gpd.read_file('../additional_data/bus_routes.json'),
    'bus_routes': unary_union(gpd.read_file('./additional_data/bus_routes.json').loc[:, 'geometry'].to_list()),
    'subway': gpd.read_file('./additional_data/subway.json'),
    'railways': gpd.read_file('./additional_data/railways.json'),
}
def calculate_min_distance_to_lines(point: Point, lines: gpd.GeoDataFrame):
    # return min(map(point.distance, lines.loc[:, 'geometry']))
    if isinstance(lines, gpd.GeoDataFrame):
        lines = MultiLineString(lines.loc[:, 'geometry'].to_list())
    return point.distance(lines)

def create_distance_to_lines_features(data: pd.DataFrame):
    features = data
    for name, lines in feature_lines.items():
        distances = []
        for _, row in tqdm(data.iterrows(), total=data.shape[0]):
            distances.append(calculate_min_distance_to_lines(Point(row['longitude'], row['latitude']), lines))
        features[f'distance_{name}'] = distances

    return pd.DataFrame(features)

settings = {
    'fill_limit': 30,
    'window_sizes': [60, 300, 600],
    'window_center': True,
    # 'window_functions': ['mean', 'std'],
    'columns': None, #['distance_London', 'distance_Brighton', 'distance_parks', 'bus_stops'],
    'functions': ['mean', 'std', 'median'],
}
shift_settings = {
    'periods': [60, 300, 600],
    'columns_patterns': ['window_'],
    'fill_limit': 30,
}




print('-------------------------validation get additional features------------------------')
data_path_valid_1 = '/home/jnli/SHL_2023/SHL2023/data2023/validate/Bag'
data_processed_validate_1 = pd.read_csv('{}/get_road_feature.csv'.format(data_path_valid_1))
data_processed_validate_1 = create_distance_to_lines_features(create_point_distance_features(data_processed_validate_1 ))
data_processed_validate_1.to_csv('{}/get_road_feature_1.csv'.format(data_path_valid_1), index=False)
print('valid_data 1 save well')


# data_path_valid_2 = '/home/jnli/SHL_2023/SHL2023/data2023/validate/Hand'
# data_processed_validate_2 = pd.read_csv('{}/get_road_feature.csv'.format(data_path_valid_2))
# data_processed_validate_2 = create_distance_to_lines_features(create_point_distance_features(data_processed_validate_2 ))
# data_processed_validate_2.to_csv('{}/get_road_feature_1.csv'.format(data_path_valid_2), index=False)
# print('valid_data 2 save well')

data_path_valid_3 = '/home/jnli/SHL_2023/SHL2023/data2023/validate/Hips'
data_processed_validate_3 = pd.read_csv('{}/get_road_feature.csv'.format(data_path_valid_3))
data_processed_validate_3 = create_distance_to_lines_features(create_point_distance_features(data_processed_validate_3 ))
data_processed_validate_3.to_csv('{}/get_road_feature_1.csv'.format(data_path_valid_3), index=False)
print('valid_data 3 save well')



data_path_valid_4 = '/home/jnli/SHL_2023/SHL2023/data2023/validate/Torso'
data_processed_validate_4 = pd.read_csv('{}/get_road_feature.csv'.format(data_path_valid_4))
print('data load well')
data_processed_validate_4 = create_distance_to_lines_features(create_point_distance_features(data_processed_validate_4 ))
data_processed_validate_4.to_csv('{}/get_road_feature_1.csv'.format(data_path_valid_4), index=False)
print('valid_data 4 save well')


# print('-------------------------train get additional features------------------------')


# # data_path_train_1 =  '/home/jnli/SHL_2023/SHL2023/data2023/train/Bag'
# # data_processed_train_1 = pd.read_csv('{}/get_road_feature.csv'.format(data_path_train_1))
# # data_processed_train_1 = create_distance_to_lines_features(create_point_distance_features(data_processed_train_1))
# # data_processed_train_1.to_csv('{}/get_road_feature_1.csv'.format(data_path_train_1), index=False)
# # print('train_data save well')


# data_path_train_2 =  '/home/jnli/SHL_2023/SHL2023/data2023/train/Hand'
# data_processed_train_2 = pd.read_csv('{}/get_road_feature.csv'.format(data_path_train_2))
# data_processed_train_2 = create_distance_to_lines_features(create_point_distance_features(data_processed_train_2))
# data_processed_train_2.to_csv('{}/get_road_feature_1.csv'.format(data_path_train_2), index=False)
# print('train_data 2 save well')



# data_path_train_3 =  '/home/jnli/SHL_2023/SHL2023/data2023/train/Hips'
# data_processed_train_3 = pd.read_csv('{}/get_road_feature.csv'.format(data_path_train_3))
# data_processed_train_3 = create_distance_to_lines_features(create_point_distance_features(data_processed_train_3))
# data_processed_train_3.to_csv('{}/get_road_feature_1.csv'.format(data_path_train_3), index=False)
# print('train_data 3 save well')

# data_path_train_4 =  '/home/jnli/SHL_2023/SHL2023/data2023/train/Torso'
# data_processed_train_4 = pd.read_csv('{}/get_road_feature.csv'.format(data_path_train_4))
# data_processed_train_4 = create_distance_to_lines_features(create_point_distance_features(data_processed_train_4))
# data_processed_train_4.to_csv('{}/get_road_feature_1.csv'.format(data_path_train_4), index=False)
# print('train_data 4 save well')


# print('-------------------------test get additional features------------------------')
# data_path_test = '/home/jnli/SHL_2023/SHL2023/data2023/test'
# data_processed_test = pd.read_csv('{}/get_road_feature.csv'.format(data_path_test))
# data_processed_test = create_distance_to_lines_features(create_point_distance_features(data_processed_test))
# data_processed_test.to_csv('{}/get_road_feature_1.csv'.format(data_processed_test), index=False)
# print('test_data save well')
