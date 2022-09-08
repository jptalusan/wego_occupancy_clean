#!/usr/bin/env python
# coding: utf-8

# In[165]:

# In[166]:

import os
import sys
sys.path.insert(0,'..')
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error
from src.config import *
from pandas.core.common import SettingWithCopyWarning
from src import data_utils, triplevel_utils
from copy import deepcopy
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
pd.set_option('display.max_columns', None)
import xgboost as xgb
import importlib
importlib.reload(data_utils)

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold


def get_mse(x, length=None):
    if not length:
        y_true = [int(float(y)) for y in x.y_true]
        y_pred = [int(float(y)) for y in x.y_pred]
    else:
        if len(x.y_true) < length:
            length = len(x.y_true)
        y_true = [int(float(y)) for y in x.y_true[0:length]]
        y_pred = [int(float(y)) for y in x.y_pred[0:length]]
        
    return mean_squared_error(y_true, y_pred)


# Define evaluator, requires some time information (merge testX, testY and pred)
def generate_results_over_time_window(results_df_arr):
    mse_df = pd.DataFrame()
    for i, results_df in enumerate(results_df_arr):
        tdf = results_df[['time_window', 'y_pred', 'y_true']].groupby('time_window').agg({'y_pred': list, 'y_true': list})
        tdf['mse'] = tdf.apply(lambda x: mean_squared_error(x.y_true, x.y_pred, squared=False), axis=1)
        mse_df[i] = tdf['mse']
    return mse_df

# In[31]:

## Day Ahead

# In[161]:

print("Starting day ahead evaluation.")
processed_triplevel = os.path.join('../data', 'processed', 'triplevel_df.parquet')
df = pd.read_parquet(processed_triplevel, engine='auto')
df = df.dropna()
# Removing time_window in case a different one will be used
df = df.drop(['time_window', 'load'], axis=1)
df = df.reset_index(drop=True)
df = df.sort_values(['block_abbr', 'transit_date', 'arrival_time', 'route_id_direction'])

FOLDS = 3
RANDOM_SEED = 100
WINDOW = 30
PAST_TRIPS = 5
TARGET = 'y_reg100'

# cat_features = ['route_id_direction', 'is_holiday', 'dayofweek', 'is_school_break', 'time_window']
cat_features = ['route_id_direction', 'is_holiday', 'dayofweek', 'is_school_break']
ord_features = ['year', 'month', 'hour', 'day', 'time_window']
num_features = ['temperature', 'humidity', 'precipitation_intensity', 'avg_sched_headway', 'traffic_speed',
                'load_pct_change', 'act_headway_pct_change', 'avg_past_act_headway', 'avg_past_trips_loads']

tdf = triplevel_utils.generate_new_day_ahead_features(df, time_window=WINDOW, past_trips=PAST_TRIPS, target=TARGET)
tdf = tdf.groupby(['transit_date', 'route_id_direction', 'time_window']).agg({"year":"first", 
                                                                              "month":"first",
                                                                              "day": "first",
                                                                              "hour":"first",
                                                                              "is_holiday": "first",
                                                                              "is_school_break": "first",
                                                                              "dayofweek":"first",
                                                                              "temperature":"mean", 
                                                                              "humidity":"mean",
                                                                              "precipitation_intensity": "mean",
                                                                              "traffic_speed":"mean",
                                                                              "scheduled_headway": "max",
                                                                              "load_pct_change": "max",
                                                                              "act_headway_pct_change": "max",
                                                                              "avg_past_act_headway": "max",
                                                                              "avg_past_trips_loads": "max",
                                                                              TARGET: "max" })
tdf = tdf.reset_index(level=[0,1,2])
print("ohe_encoder is for the following column order:", cat_features)
rf_df, ix_map, ohe_encoder, percentiles = triplevel_utils.prepare_df_for_training(tdf, cat_features, ord_features, target=TARGET)
rf_df, percentiles = triplevel_utils.adjust_bins(rf_df, TARGET=TARGET, percentiles=percentiles)

target_load = rf_df[[TARGET]]

original_rf = deepcopy(rf_df)
original_rf['time_window'] = tdf['time_window']
drop_cols = ['time_window', 'route_id', 'route_direction_name', 'block_abbr', 'y_reg100', 'y_reg095', 'transit_date', 'is_holiday', 'route_id_direction', 'actual_headways', 'trip_id', 'arrival_time']
drop_cols = [col for col in drop_cols if col in rf_df.columns]
rf_df = rf_df.drop(drop_cols, axis=1)

y = rf_df.pop('y_class')
X = rf_df

# In[162]:


# # Grid search results
# fp = os.path.join('../models', 'day_ahead', 'XGBOOST_RANDSEARCHCV_day_ahead_with_schoolbreak012.joblib')
# search_results = joblib.load(fp)
# print(search_results.best_params_)

# In[163]:


columns = rf_df.columns

# n_estimators  = search_results.best_params_['n_estimators']
# max_depth     = search_results.best_params_['max_depth']
# learning_rate = search_results.best_params_['learning_rate']
# gamma         = search_results.best_params_['gamma']
objective = 'multi:softmax'

skf = StratifiedKFold(n_splits=FOLDS, random_state=RANDOM_SEED, shuffle=True)
skf.get_n_splits(X, y)

results_df_arr = []
kfold = 0
for train_index, test_index in skf.split(X, y):
    _X_train, _X_test = X.iloc[train_index], X.iloc[test_index]
    _y_train, _y_test = y.iloc[train_index], y.iloc[test_index]
    
    # model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
    #                           learning_rate=learning_rate, use_label_encoder=False, 
    #                           gamma=gamma, num_class=3,
    #                           objective=objective, eval_metric='mlogloss')
    model = xgb.XGBClassifier(use_label_encoder=False, objective=objective, eval_metric='mlogloss', num_class=3)
    model.fit(_X_train, _y_train, verbose=1)

    preds = model.predict(_X_test)
    _X_test = pd.DataFrame(_X_test, columns=columns)
    _X_test['y_pred'] = preds
    _X_test['y_true'] = _y_test
    res_df = _X_test[['y_true', 'y_pred']]
    res_df['time_window'] = original_rf.iloc[test_index]['time_window'].tolist()
    res_df['kfold'] = kfold
    kfold = kfold + 1
    results_df_arr.append(res_df)

fp = os.path.join('../evaluation', 'day_ahead_012_raw_results_notOHE_tw.pkl')
pd.concat(results_df_arr).to_pickle(fp)

mse_df = generate_results_over_time_window(results_df_arr)
fp = os.path.join('../evaluation', 'day_ahead_012_results_notOHE_tw.pkl')
mse_df.to_pickle(fp)

# In[164]:


# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)

# fig, ax = plt.subplots(figsize=(10, 5))
# mse_df['time_window'] = mse_df.index
# mse_df['time_window'].to_numpy()
# mse_df['time_window'] = mse_df['time_window'].to_numpy().astype('int')
# mse_df = mse_df.reset_index(drop=True).set_index('time_window')

# mse_df.T.boxplot(ax=ax)

# ax.set_title("Day Ahead")
# ax.set_ylabel("Root Mean Square Error")
# ax.set_xlabel("30 minute time windows")
# ax.set_ylim(0.0, 2.5)

# fp = os.path.join('../plots', 'day_ahead_012.png')
# plt.savefig(fp, dpi=200, bbox_inches='tight')

# # In[99]:


# rf_df, ix_map, ohe_encoder, percentiles = triplevel_utils.prepare_df_for_training(tdf, cat_features, ord_features, target=TARGET)
# rf_df, percentiles = triplevel_utils.adjust_bins(rf_df, TARGET=TARGET, percentiles=percentiles)

# original_rf = deepcopy(rf_df)
# original_rf['time_window'] = tdf['time_window']
# drop_cols = ['time_window', 'route_id', 'route_direction_name', 'block_abbr', 'y_reg100', 'y_reg095', 'transit_date', 'is_holiday', 'route_id_direction']
# drop_cols = [col for col in drop_cols if col in rf_df.columns]
# rf_df = rf_df.drop(drop_cols, axis=1)
# rf_df = rf_df[rf_df['y_class'] >= 2]

# y = rf_df.pop('y_class')
# y = y - 2
# X = rf_df

# # In[ ]:


# # # Grid search results
# fp = os.path.join('../models', 'day_ahead', 'XGBOOST_RANDSEARCHCV_day_ahead_with_schoolbreak234.joblib')
# search_results = joblib.load(fp)
# print(search_results.best_params_)

# # In[100]:


# columns = rf_df.columns

# n_estimators  = search_results.best_params_['n_estimators']
# max_depth     = search_results.best_params_['max_depth']
# learning_rate = search_results.best_params_['learning_rate']
# gamma         = search_results.best_params_['gamma']
# objective = 'multi:softmax'

# skf = StratifiedKFold(n_splits=FOLDS, random_state=RANDOM_SEED, shuffle=True)
# skf.get_n_splits(X, y)

# results_df_arr = []
# kfold = 0
# for train_index, test_index in skf.split(X, y):
#     _X_train, _X_test = X.iloc[train_index], X.iloc[test_index]
#     _y_train, _y_test = y.iloc[train_index], y.iloc[test_index]
    
#     # model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
#     #                           learning_rate=learning_rate, use_label_encoder=False, 
#     #                           gamma=gamma, num_class=3,
#     #                           objective=objective, eval_metric='mlogloss')
#     model = xgb.XGBClassifier(use_label_encoder=False, objective=objective, eval_metric='mlogloss', num_class=3)
#     model.fit(_X_train, _y_train, verbose=1)

#     preds = model.predict(_X_test)
#     _X_test = pd.DataFrame(_X_test, columns=columns)
#     _X_test['y_pred'] = preds
#     _X_test['y_true'] = _y_test
#     res_df = _X_test[['y_true', 'y_pred']]
#     res_df['time_window'] = original_rf.iloc[test_index]['time_window'].tolist()
#     res_df['kfold'] = kfold
#     kfold = kfold + 1
#     results_df_arr.append(res_df)

# fp = os.path.join('../evaluation', 'day_ahead_234_raw_results.pkl')
# pd.concat(results_df_arr).to_pickle(fp)

# mse_df = generate_results_over_time_window(results_df_arr)
# fp = os.path.join('../evaluation', 'day_ahead_234_results_schoolbreak.pkl')
# mse_df.to_pickle(fp)

# fig, ax = plt.subplots(figsize=(10, 5))
# mse_df['time_window'] = mse_df.index
# mse_df['time_window'].to_numpy()
# mse_df['time_window'] = mse_df['time_window'].to_numpy().astype('int')
# mse_df = mse_df.reset_index(drop=True).set_index('time_window')

# mse_df.T.boxplot(ax=ax)

# ax.set_title("Day Ahead, bins: 234")
# ax.set_ylabel("Root Mean Square Error")
# ax.set_xlabel("30 minute time windows")
# ax.set_ylim(0.0, 2.5)

# fp = os.path.join('../plots', 'day_ahead_234.png')
# plt.savefig(fp, dpi=200, bbox_inches='tight')

## Any Ahead

# In[101]:
print("Starting any day evaluation.")


processed_triplevel = os.path.join('../data', 'processed', 'triplevel_df.parquet')
df = pd.read_parquet(processed_triplevel, engine='auto')
df = df.dropna()
# Removing time_window in case a different one will be used
df = df.drop(['time_window', 'load'], axis=1)
df = df.reset_index(drop=True)
df = df.sort_values(['block_abbr', 'transit_date', 'arrival_time', 'route_id_direction'])

FOLDS = 3
RANDOM_SEED = 100
WINDOW = 30
PAST_TRIPS = 5
TARGET = 'y_reg100'

# cat_features = ['route_id_direction', 'is_holiday', 'dayofweek', 'is_school_break', 'time_window']
cat_features = ['route_id_direction', 'is_holiday', 'dayofweek', 'is_school_break']
ord_features = ['year', 'month', 'hour', 'day', 'time_window']
num_features = ['temperature', 'humidity', 'precipitation_intensity', 'avg_sched_headway', 'traffic_speed']

tdf = triplevel_utils.generate_new_day_ahead_features(df, time_window=WINDOW, past_trips=PAST_TRIPS, target=TARGET)
tdf = tdf.groupby(['transit_date', 'route_id_direction', 'time_window']).agg({"year":"first", 
                                                                              "month":"first",
                                                                              "day": "first",
                                                                              "hour":"first",
                                                                              "is_holiday": "first",
                                                                              "is_school_break": "first",
                                                                              "dayofweek":"first",
                                                                              "temperature":"mean", 
                                                                              "humidity":"mean",
                                                                              "precipitation_intensity": "mean",
                                                                              "traffic_speed":"mean",
                                                                              "scheduled_headway": "max",
                                                                              TARGET: "max" })
tdf = tdf.reset_index(level=[0,1,2])
print("ohe_encoder is for the following column order:", cat_features)
rf_df, ix_map, ohe_encoder, percentiles = triplevel_utils.prepare_df_for_training(tdf, cat_features, ord_features, target=TARGET)
rf_df, percentiles = triplevel_utils.adjust_bins(rf_df, TARGET=TARGET, percentiles=percentiles)

target_load = rf_df[[TARGET]]
original_rf = deepcopy(rf_df)
original_rf['time_window'] = tdf['time_window']
drop_cols = ['time_window', 'route_id', 'route_direction_name', 'block_abbr', 'y_reg100', 'y_reg095', 'transit_date', 'is_holiday', 'route_id_direction', 'actual_headways', 'trip_id', 'arrival_time']
drop_cols = [col for col in drop_cols if col in rf_df.columns]
rf_df = rf_df.drop(drop_cols, axis=1)

y = rf_df.pop('y_class')
X = rf_df

# In[102]:


# Grid search results
# fp = os.path.join('../models', 'any_day', 'XGBOOST_RANDSEARCHCV_any_day_with_schoolbreak012.joblib')
# search_results = joblib.load(fp)
# print(search_results.best_params_)

# In[103]:

columns = rf_df.columns

# n_estimators  = search_results.best_params_['n_estimators']
# max_depth     = search_results.best_params_['max_depth']
# learning_rate = search_results.best_params_['learning_rate']
# gamma         = search_results.best_params_['gamma']
objective = 'multi:softmax'

skf = StratifiedKFold(n_splits=FOLDS, random_state=RANDOM_SEED, shuffle=True)
skf.get_n_splits(X, y)

results_df_arr = []
kfold = 0
for train_index, test_index in skf.split(X, y):
    _X_train, _X_test = X.iloc[train_index], X.iloc[test_index]
    _y_train, _y_test = y.iloc[train_index], y.iloc[test_index]
    
    # model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
    #                           learning_rate=learning_rate, use_label_encoder=False, 
    #                           gamma=gamma, num_class=3,
    #                           objective=objective, eval_metric='mlogloss')
    model = xgb.XGBClassifier(use_label_encoder=False, objective=objective, eval_metric='mlogloss', num_class=3)
    model.fit(_X_train, _y_train, verbose=1)

    preds = model.predict(_X_test)
    _X_test = pd.DataFrame(_X_test, columns=columns)
    _X_test['y_pred'] = preds
    _X_test['y_true'] = _y_test
    res_df = _X_test[['y_true', 'y_pred']]
    res_df['time_window'] = original_rf.iloc[test_index]['time_window'].tolist()
    res_df['kfold'] = kfold
    kfold = kfold + 1
    results_df_arr.append(res_df)

fp = os.path.join('../evaluation', 'any_day_012_raw_results_notOHE_tw.pkl')
pd.concat(results_df_arr).to_pickle(fp)

mse_df = generate_results_over_time_window(results_df_arr)
fp = os.path.join('../evaluation', 'any_day_012_results_notOHE_tw.pkl')
mse_df.to_pickle(fp)

# In[104]:


# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)

# fig, ax = plt.subplots(figsize=(10, 5))
# mse_df['time_window'] = mse_df.index
# mse_df['time_window'].to_numpy()
# mse_df['time_window'] = mse_df['time_window'].to_numpy().astype('int')
# mse_df = mse_df.reset_index(drop=True).set_index('time_window')

# mse_df.T.boxplot(ax=ax)

# ax.set_title("Any Day, bins:012")
# ax.set_ylabel("Root Mean Square Error")
# ax.set_xlabel("30 minute time windows")
# ax.set_ylim(0.0, 2.5)

# fp = os.path.join('../plots', 'any_day_012.png')
# plt.savefig(fp, dpi=200, bbox_inches='tight')

# # In[112]:


# rf_df, ix_map, ohe_encoder, percentiles = triplevel_utils.prepare_df_for_training(tdf, cat_features, ord_features, target=TARGET)
# rf_df, percentiles = triplevel_utils.adjust_bins(rf_df, TARGET=TARGET, percentiles=percentiles)
# original_rf = deepcopy(rf_df)
# original_rf['time_window'] = tdf['time_window']
# drop_cols = ['time_window', 'route_id', 'route_direction_name', 'block_abbr', 'y_reg100', 'y_reg095', 'transit_date', 'is_holiday', 'route_id_direction']
# drop_cols = [col for col in drop_cols if col in rf_df.columns]
# rf_df = rf_df.drop(drop_cols, axis=1)
# rf_df = rf_df[rf_df['y_class'] >= 2]

# y = rf_df.pop('y_class')
# y = y - 2
# X = rf_df

# # In[ ]:


# # Grid search results
# fp = os.path.join('../models', 'any_day', 'XGBOOST_RANDSEARCHCV_any_day_with_schoolbreak234.joblib')
# search_results = joblib.load(fp)
# print(search_results.best_params_)

# # In[115]:


# columns = rf_df.columns

# n_estimators  = search_results.best_params_['n_estimators']
# max_depth     = search_results.best_params_['max_depth']
# learning_rate = search_results.best_params_['learning_rate']
# gamma         = search_results.best_params_['gamma']
# objective = 'multi:softmax'

# skf = StratifiedKFold(n_splits=FOLDS, random_state=RANDOM_SEED, shuffle=True)
# skf.get_n_splits(X, y)

# results_df_arr = []
# kfold = 0
# for train_index, test_index in skf.split(X, y):
#     _X_train, _X_test = X.iloc[train_index], X.iloc[test_index]
#     _y_train, _y_test = y.iloc[train_index], y.iloc[test_index]
    
#     # model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
#     #                           learning_rate=learning_rate, use_label_encoder=False, 
#     #                           gamma=gamma, num_class=3,
#     #                           objective=objective, eval_metric='mlogloss')
#     model = xgb.XGBClassifier(use_label_encoder=False, objective=objective, eval_metric='mlogloss', num_class=3)
#     model.fit(_X_train, _y_train, verbose=1)

#     preds = model.predict(_X_test)
#     _X_test = pd.DataFrame(_X_test, columns=columns)
#     _X_test['y_pred'] = preds
#     _X_test['y_true'] = _y_test
#     res_df = _X_test[['y_true', 'y_pred']]
#     res_df['time_window'] = original_rf.iloc[test_index]['time_window'].tolist()
#     res_df['kfold'] = kfold
#     kfold = kfold + 1
#     results_df_arr.append(res_df)

# fp = os.path.join('../evaluation', 'any_day_234_raw_results.pkl')
# pd.concat(results_df_arr).to_pickle(fp)

# mse_df = generate_results_over_time_window(results_df_arr)
# fp = os.path.join('../evaluation', 'any_day_234_results_schoolbreak.pkl')
# mse_df.to_pickle(fp)

# fig, ax = plt.subplots(figsize=(10, 5))
# mse_df['time_window'] = mse_df.index
# mse_df['time_window'].to_numpy()
# mse_df['time_window'] = mse_df['time_window'].to_numpy().astype('int')
# mse_df = mse_df.reset_index(drop=True).set_index('time_window')

# mse_df.T.boxplot(ax=ax)

# ax.set_title("Any Day, bins: 234")
# ax.set_ylabel("Root Mean Square Error")
# ax.set_xlabel("30 minute time windows")
# ax.set_ylim(0.0, 2.5)

# fp = os.path.join('../plots', 'any_day_234.png')
# plt.savefig(fp, dpi=200, bbox_inches='tight')

# ## Baseline

# In[119]:


import numpy as np
import swifter

# In[144]:


# def get_statistical_prediction(row, percentile, lookback_duration, TARGET='y_reg100'):
#     trip_id = row.trip_id
#     transit_date = row.transit_date
#     route_id_direction = row.route_id_direction
#     lookback_date = transit_date - pd.Timedelta(lookback_duration)
#     tdf = df[(df['transit_date'] >= lookback_date) & \
#              (df['transit_date'] < transit_date)]
#     tdf = tdf[(tdf['trip_id'] == trip_id) & \
#               (tdf['route_id_direction'] == route_id_direction)]
#     if tdf.empty:
#         return -1
#     return np.percentile(tdf[TARGET].to_numpy(), percentile)

# # In[145]:


# processed_triplevel = os.path.join('../data', 'processed', 'triplevel_df.parquet')
# df = pd.read_parquet(processed_triplevel, engine='auto')
# df = df.dropna()
# # Removing time_window in case a different one will be used
# df = df.drop(['time_window', 'load'], axis=1)
# df = df.reset_index(drop=True)
# df = df.sort_values(['block_abbr', 'transit_date', 'arrival_time', 'route_id_direction'])

# FOLDS = 3
# RANDOM_SEED = 100
# WINDOW = 30
# PAST_TRIPS = 5
# TARGET = 'y_reg100'

# percentiles = [(0.0, 9.0), (10.0, 16.0), (17.0, 55.0), (56.0, 75.0), (76.0, 100.0)]
# df['y_class'] = df[TARGET].swifter.apply(lambda x: data_utils.get_class(x, percentiles))
# df['y_class'] = df['y_class'].astype('int')

# df['minute'] = df['arrival_time'].dt.minute
# df['minuteByWindow'] = df['minute'] // WINDOW
# df['temp'] = df['minuteByWindow'] + (df['hour'] * 60 / WINDOW)
# df['time_window'] = np.floor(df['temp']).astype('int')
# df = df.drop(columns=['minute', 'minuteByWindow', 'temp'], axis=1)
# df.head()

# # In[ ]:


# skf = StratifiedKFold(n_splits=FOLDS, random_state=RANDOM_SEED, shuffle=True)
# X, y = df[['transit_date', 'trip_id', 'arrival_time', 'route_id_direction', 'time_window', TARGET]], df['y_class']
# skf.get_n_splits(X, y)

# lookback_distances = ['4W', '2W', '1W']
# percentile = 1.0
# results_df_arr = []
# for _, test_index in skf.split(X, y):
#     for lookback_distance in lookback_distances:
#         baseline_X = X.iloc[test_index]
#         baseline_Y = y.iloc[test_index]
        
#         baseline_X['y_pred'] = baseline_X.swifter.apply(lambda x: get_statistical_prediction(x, percentile, lookback_distance, TARGET=TARGET), axis=1)
#         baseline_X['y_true'] = baseline_Y.to_numpy()
#         res_df = baseline_X[['time_window', 'y_true', 'y_pred']]
#         results_df_arr.append(res_df)
#     break

# # In[152]:


# results_df_arr[0]['y_pred_class'] = results_df_arr[0]['y_pred'].apply(lambda x: data_utils.get_class(x, percentiles))
# results_df_arr[1]['y_pred_class'] = results_df_arr[1]['y_pred'].apply(lambda x: data_utils.get_class(x, percentiles))
# results_df_arr[2]['y_pred_class'] = results_df_arr[2]['y_pred'].apply(lambda x: data_utils.get_class(x, percentiles))
# df1 = results_df_arr[0].dropna()
# df2 = results_df_arr[1].dropna()
# df3 = results_df_arr[2].dropna()
# df1 = df1[['time_window', 'y_true', 'y_pred_class']]
# df2 = df2[['time_window', 'y_true', 'y_pred_class']]
# df3 = df3[['time_window', 'y_true', 'y_pred_class']]
# df1 = df1.rename(columns={'y_pred_class': 'y_pred'})
# df2 = df2.rename(columns={'y_pred_class': 'y_pred'})
# df3 = df3.rename(columns={'y_pred_class': 'y_pred'})
# df1['past'] = 1
# df2['past'] = 2
# df3['past'] = 4

# fp = os.path.join('../evaluation', 'baseline_012_raw_results.pkl')
# pd.concat([df1, df2, df3]).to_pickle(fp)

# mse_df = generate_results_over_time_window([df1, df2, df3])
# fp = os.path.join('../evaluation', 'baseline_012_results.pkl')
# mse_df.to_pickle(fp)

# # In[155]:


# fig, ax = plt.subplots(figsize=(10, 5))
# mse_df['time_window'] = mse_df.index
# mse_df['time_window'].to_numpy()
# mse_df['time_window'] = mse_df['time_window'].to_numpy().astype('int')
# mse_df = mse_df.reset_index(drop=True).set_index('time_window')

# mse_df.T.boxplot(ax=ax)

# ax.set_title("Baseline")
# ax.set_ylabel("Root Mean Square Error")
# ax.set_xlabel("30 minute time windows")
# ax.set_ylim(0.0, 2.5)

# fp = os.path.join('../plots', 'day_ahead_baseline_012.png')
# plt.savefig(fp, dpi=200, bbox_inches='tight')

# ### Baseline 234

# # In[157]:


# processed_triplevel = os.path.join('../data', 'processed', 'triplevel_df.parquet')
# df = pd.read_parquet(processed_triplevel, engine='auto')
# df = df.dropna()
# # Removing time_window in case a different one will be used
# df = df.drop(['time_window', 'load'], axis=1)
# df = df.reset_index(drop=True)
# df = df.sort_values(['block_abbr', 'transit_date', 'arrival_time', 'route_id_direction'])

# FOLDS = 3
# RANDOM_SEED = 100
# WINDOW = 30
# PAST_TRIPS = 5
# TARGET = 'y_reg100'
# df = df[df[TARGET] >= 17]

# percentiles = [(16.0, 55.0), (56.0, 75.0), (76.0, 100.0)]
# df['y_class'] = df[TARGET].swifter.apply(lambda x: data_utils.get_class(x, percentiles))
# df['y_class'] = df['y_class'].astype('int')

# df['minute'] = df['arrival_time'].dt.minute
# df['minuteByWindow'] = df['minute'] // WINDOW
# df['temp'] = df['minuteByWindow'] + (df['hour'] * 60 / WINDOW)
# df['time_window'] = np.floor(df['temp']).astype('int')
# df = df.drop(columns=['minute', 'minuteByWindow', 'temp'], axis=1)
# df.head()

# # In[158]:


# skf = StratifiedKFold(n_splits=FOLDS, random_state=RANDOM_SEED, shuffle=True)
# X, y = df[['transit_date', 'trip_id', 'arrival_time', 'route_id_direction', 'time_window', TARGET]], df['y_class']
# skf.get_n_splits(X, y)

# lookback_distances = ['4W', '2W', '1W']
# percentile = 1.0
# results_df_arr = []
# for _, test_index in skf.split(X, y):
#     for lookback_distance in lookback_distances:
#         baseline_X = X.iloc[test_index]
#         baseline_Y = y.iloc[test_index]
        
#         baseline_X['y_pred'] = baseline_X.swifter.apply(lambda x: get_statistical_prediction(x, percentile, lookback_distance, TARGET=TARGET), axis=1)
#         baseline_X['y_true'] = baseline_Y.to_numpy()
#         res_df = baseline_X[['time_window', 'y_true', 'y_pred']]
#         results_df_arr.append(res_df)
#     break

# # In[159]:


# results_df_arr[0]['y_pred_class'] = results_df_arr[0]['y_pred'].apply(lambda x: data_utils.get_class(x, percentiles))
# results_df_arr[1]['y_pred_class'] = results_df_arr[1]['y_pred'].apply(lambda x: data_utils.get_class(x, percentiles))
# results_df_arr[2]['y_pred_class'] = results_df_arr[2]['y_pred'].apply(lambda x: data_utils.get_class(x, percentiles))
# df1 = results_df_arr[0].dropna()
# df2 = results_df_arr[1].dropna()
# df3 = results_df_arr[2].dropna()
# df1 = df1[['time_window', 'y_true', 'y_pred_class']]
# df2 = df2[['time_window', 'y_true', 'y_pred_class']]
# df3 = df3[['time_window', 'y_true', 'y_pred_class']]
# df1 = df1.rename(columns={'y_pred_class': 'y_pred'})
# df2 = df2.rename(columns={'y_pred_class': 'y_pred'})
# df3 = df3.rename(columns={'y_pred_class': 'y_pred'})
# df1['past'] = 1
# df2['past'] = 2
# df3['past'] = 4

# fp = os.path.join('../evaluation', 'baseline_234_raw_results.pkl')
# pd.concat([df1, df2, df3]).to_pickle(fp)

# mse_df = generate_results_over_time_window([df1, df2, df3])
# fp = os.path.join('../evaluation', 'baseline_234_results.pkl')
# mse_df.to_pickle(fp)

# # In[160]:


# fig, ax = plt.subplots(figsize=(10, 5))
# mse_df['time_window'] = mse_df.index
# mse_df['time_window'].to_numpy()
# mse_df['time_window'] = mse_df['time_window'].to_numpy().astype('int')
# mse_df = mse_df.reset_index(drop=True).set_index('time_window')

# mse_df.T.boxplot(ax=ax)

# ax.set_title("Baseline, bins:234")
# ax.set_ylabel("Root Mean Square Error")
# ax.set_xlabel("30 minute time windows")
# ax.set_ylim(0.0, 2.5)

# fp = os.path.join('../plots', 'day_ahead_baseline_234.png')
# plt.savefig(fp, dpi=200, bbox_inches='tight')
