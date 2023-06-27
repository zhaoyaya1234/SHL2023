import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier


from utils import helper
from utils import TimeKeeper
from utils import DataLoader, DataProcesser

class Solver():
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
        print("X_train: {}".format(self.X_train.shape))
        print("y_train: {}".format(len(self.y_train)))
        

    def train(self):


        self.train_model_rf()
        self.train_model_lgb()
        self.train_model_xgb()

       
    def train_model_rf(self):
        print("Training rf...")
        timer = TimeKeeper.TimeKeeper()
        self.model_rf = RandomForestClassifier(n_estimators = 20, random_state = 0, max_depth = 8)
        self.model_rf.fit(self.X_train, self.y_train)
        print("Time elapsed for training rf: {}".format(timer.get_update_time()))

    def train_model_lgb(self):
        print("Training LightGBM...")
        timer = TimeKeeper.TimeKeeper()
        X_train, X_test, y_train, y_test = train_test_split(self.X_train, self.y_train, test_size = 0.2)
        train_data = lgb.Dataset(X_train, label = y_train - 1)
        test_data = lgb.Dataset(X_test, label = y_test - 1)
        params={
            'learning_rate':0.1,
            'lambda_l1':0.1,
            'lambda_l2':0.2,
            'max_depth':6,
            'objective':'multiclass',
            'num_class':8,  
        }
        self.model_lgb = lgb.train(params, train_data, valid_sets = [test_data])
        print("Time elapsed for training LightGBM: {}".format(timer.get_update_time()))

    def train_model_xgb(self):
        print("Training XGBoost...")
        timer = TimeKeeper.TimeKeeper()
        self.model_xgb = xgb.XGBClassifier(
        n_estimators=50,  # 减少弱学习器的数量
        max_depth=3,  # 减少决策树的最大深度
        objective="multi:softmax",
        num_class=8,
        subsample=0.8,  # 子采样比例
        colsample_bytree=0.8  ) # 特征采样比例

        self.model_xgb.fit(self.X_train, self.y_train)
        print("Time elapsed for training XGBoost: {}".format(timer.get_update_time()))

  
    def predict_raw(self, X_val = None, y_val = None):
 
        self.X_val = X_val
        print("X_val  : {}".format(self.X_val.shape))
        
        try:
            self.y_val = y_val
            print("y_val  : {}".format(len(self.y_val)))
        except:
            print('datasets  no label')



        print('predict rf')
        timer = TimeKeeper.TimeKeeper()
        self.pred_rf = self.model_rf.predict(X_val) 
        self.pred_prob_rf = self.model_rf.predict_proba(X_val) 
        print("Time elapsed for testing rf {}".format(timer.get_update_time()))


        print('predict lgb')
        timer = TimeKeeper.TimeKeeper()
        self.pred_prob_lgb = self.model_lgb.predict(X_val)
        self.pred_lgb = self.pred_prob_lgb.argmax(axis = 1) + 1
        print("Time elapsed for testing lgb: {}".format(timer.get_update_time()))

        print('predict xgb')
        timer = TimeKeeper.TimeKeeper()
        self.pred_xgb = self.model_xgb.predict(X_val)
        self.pred_prob_xgb = self.model_xgb.predict_proba(X_val)
        print("Time elapsed for testing xgb: {}".format(timer.get_update_time()))

    
        
    def predict_mean(self, window_size = 150):


        print('----------------------------rf bag lgb xgb catboost  ---------------------------------')
        pred_prob = self.pred_prob_rf +self.pred_prob_lgb +self.pred_prob_xgb
        pred_prob_mean = pd.DataFrame(pred_prob).rolling(window_size, center = True).mean().fillna(method = 'ffill').fillna(method = 'bfill')
        self.pred_mean = pd.Series(np.array(pred_prob_mean).argmax(axis = 1) + 1)
        
    
    def calculate_feature_importance(self):
        # Calculate feature importances for each model
        rf_feature_importance = self.model_rf.feature_importances_
        lgb_feature_importance = self.model_lgb.feature_importance(importance_type='gain')
        xgb_feature_importance = self.model_xgb.feature_importances_

        # Normalize feature importances
        rf_feature_importance = rf_feature_importance / np.sum(rf_feature_importance)
        lgb_feature_importance = lgb_feature_importance / np.sum(lgb_feature_importance)
        xgb_feature_importance = xgb_feature_importance / np.sum(xgb_feature_importance)

        # Combine feature importances from different models
        total_feature_importance = rf_feature_importance + lgb_feature_importance + xgb_feature_importance

        # Get the column names from X_train
        column_names = list(self.X_train.columns)

        # Create a dictionary mapping feature names to importance values
        feature_importance_dict = {column_names[i]: total_feature_importance[i] for i in range(len(column_names))}

        # Sort the dictionary by importance values in descending order
        feature_importance_dict = dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        self.feature_importance = total_feature_importance
        print('single experiment feature importance :{}'.format(feature_importance_dict))
