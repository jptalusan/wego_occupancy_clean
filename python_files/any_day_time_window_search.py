#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import os
import pandas as pd
pd.set_option('display.max_columns', None)
from copy import deepcopy
from pathlib import Path
import joblib
import xgboost as xgb
import sys
sys.path.insert(0,'..')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from pandas.core.common import SettingWithCopyWarning
from src import data_utils, triplevel_utils
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit, StratifiedKFold
import datetime as dt
import swifter
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
# Requires the preprocessed dataset `triplevel_df.parquet`

OUTPUT_DIR = os.path.join('../models', 'any_day', 'variable_timewindow')
ohe_features = ['route_id_direction', 'is_holiday', 'dayofweek', 'is_school_break', 'time_window']
ord_features = ['year', 'month', 'hour', 'day']
num_features = ['temperature', 'humidity', 'precipitation_intensity', 'avg_sched_headway', 'traffic_speed']
feature_label = 'y_class'


# In[2]:


processed_triplevel = os.path.join('../data', 'processed', 'triplevel_df.parquet')
df = pd.read_parquet(processed_triplevel, engine='auto')
df = df.dropna()
# Removing time_window in case a different one will be used
df = df.drop(['time_window', 'load'], axis=1)
df = df.reset_index(drop=True)
df = df.sort_values(['block_abbr', 'transit_date', 'arrival_time', 'route_id_direction'])
df['day'] = df.transit_date.dt.day
df['year'] = df.transit_date.dt.year
df = df[df["y_reg100"] < 100]


# In[10]:


time_windows = [1, 10, 20, 30, 40, 50, 60, 120]
percentiles = [(0.0, 9.0), (10.0, 16.0), (17.0, 55.0), (56.0, 75.0), (76.0, 100.0)]
# tdf = triplevel_utils.generate_new_features(df, time_window=config.time_window, past_trips=config.past_trips, target=config.target)


# In[11]:


def reconstruct_original_data(df, ix_map, ohe_encoder):
    df[ord_features] = ohe_encoder.inverse_transform(df.filter(regex='route_id_direction_|is_holiday_|dayofweek_|is_school_break_|time_window_'))
    
    for col in ord_features:
        inv_map = {v: k for k, v in ix_map[col].items()}
        df[col] = df[f"{col}_ix"].apply(lambda x: inv_map[x])
        
    df = df.drop(columns=df.filter(regex='route_id_direction_|is_holiday_|dayofweek_|is_school_break_|time_window_|_ix').columns, axis=1)
    return df


# In[12]:


ix_map = {}
for col in ord_features:
    ix_map[col] = triplevel_utils.create_ix_map(df, df, col)
    df[f"{col}_ix"] = df[col].apply(lambda x: ix_map[col][x])
# df = df.drop(columns=ord_features)
df['y_class'] = df['y_reg100'].apply(lambda x: data_utils.get_class(x, percentiles)) 
df.head(1)


# In[13]:


objective = 'multi:softmax'
    

for tw in time_windows:
    # sss = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=100)
    sss = StratifiedKFold(n_splits=3, random_state=100, shuffle=True)

    sss.get_n_splits(df)
    
    # columns = X.columns
    
    reconstructed_df_arr = []
    kfold = 0
    for train_index, test_index in sss.split(df, df['y_class'].to_numpy()):
        # y_train, y_test = df['y_class'].iloc[train_index], df['y_class'].iloc[test_index]
        
        df = triplevel_utils.generate_new_features(df, time_window=tw, target='y_reg100')
        # test = triplevel_utils.generate_new_features(test, time_window=tw, target='y_reg100')
        
        ohe_encoder = OneHotEncoder()
        ohe_encoder = ohe_encoder.fit(df[ohe_features])
        
        train, test = df.iloc[train_index], df.iloc[test_index]
        # TRAINING
        train = train.groupby(['transit_date', 'route_id_direction', 'time_window']).agg({"trip_id":"first",
                                                                                    "year_ix":"first", 
                                                                                    "month_ix":"first",
                                                                                    "day_ix": "first",
                                                                                    "hour_ix":"first",
                                                                                    "is_holiday": "first",
                                                                                    "is_school_break": "first",
                                                                                    "dayofweek":"first",
                                                                                    "temperature":"mean", 
                                                                                    "humidity":"mean",
                                                                                    "precipitation_intensity": "mean",
                                                                                    "traffic_speed":"mean",
                                                                                    "scheduled_headway": "max",
                                                                                    "y_reg100": "max" })
        train = train.reset_index(level=[0,1,2])
        train[ohe_encoder.get_feature_names_out()] = ohe_encoder.transform(train[ohe_features]).toarray()
        train = train.drop(columns=ohe_features, axis=1)
        
        drop_cols = ['time_window', 'route_id', 'route_direction_name', 'block_abbr', 'y_reg100', 'y_reg095', 'transit_date', 'is_holiday', 'route_id_direction', 'actual_headways', 'trip_id', 'arrival_time']
        
        train['y_class'] = train['y_reg100'].apply(lambda x: data_utils.get_class(x, percentiles))
        
        drop_cols = [col for col in drop_cols if col in train.columns]
        rf_df = train.drop(drop_cols, axis=1)
        
        y_train = rf_df.pop('y_class')
        X_train = rf_df
        train_columns = X_train.columns
        model = xgb.XGBClassifier(use_label_encoder=False, 
                                  objective=objective, 
                                  eval_metric='mlogloss', 
                                  num_class=5)

        model.fit(X_train, y_train, verbose=1)

        # TESTING
        test = test[test["y_reg100"] < 100]
        test['y_class'] = test['y_reg100'].apply(lambda x: data_utils.get_class(x, percentiles)) 
        
        test = test.drop(columns=ord_features, axis=1)
        test[ohe_encoder.get_feature_names_out()] = ohe_encoder.transform(test[ohe_features]).toarray()
        test = test.drop(columns=ohe_features, axis=1)
        
        drop_cols = ['time_window', 'route_id', 'route_direction_name', 'block_abbr', 'y_reg100', 'y_reg095', 'transit_date', 'is_holiday', 'route_id_direction', 'actual_headways', 'trip_id', 'arrival_time']
        drop_cols = [col for col in drop_cols if col in test.columns]
        test = test.drop(drop_cols, axis=1)
        
        y_test = test.pop("y_class")
        X_test = test[train_columns]
        
        y_pred = model.predict(X_test)
        _original_rf = deepcopy(df.iloc[test_index])
        _original_rf['y_pred'] = y_pred
        _original_rf['y_true'] = y_test
        _original_rf['kfold'] = kfold
        _original_rf['time_window_param'] = tw
        kfold = kfold + 1
        reconstructed_df_arr.append(_original_rf)
        fp = f'/home/jptalusan/mta_stationing_problem/evaluation/any_day_time_windows_fixed/{tw}_raw_res.pkl'
        _original_rf.to_pickle(fp)


# In[ ]:


reconstructed_df_arr = pd.concat(reconstructed_df_arr)
fp = f'/home/jptalusan/mta_stationing_problem/evaluation/any_day_time_windows_fixed/all_concat.pkl'
reconstructed_df_arr.to_pickle(fp)


# In[ ]:




