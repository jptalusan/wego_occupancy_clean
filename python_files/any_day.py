#!/usr/bin/env python
# coding: utf-8

# # Imports and setup

# In[1]:


import os
import sys
sys.path.insert(0,'..')


# In[6]:


from copy import deepcopy
from src.config import *
from pandas.core.common import SettingWithCopyWarning
from src import data_utils, triplevel_utils
from pyspark.sql import SparkSession

import numpy as np
import datetime as dt
import seaborn as sns
import joblib

import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
pd.set_option('display.max_columns', None)
import xgboost as xgb


# In[7]:


import importlib
importlib.reload(data_utils)


# In[8]:


spark = SparkSession.builder.config('spark.executor.cores', '8').config('spark.executor.memory', '40g')\
        .config("spark.sql.session.timeZone", "UTC").config('spark.driver.memory', '20g').master("local[26]")\
        .appName("wego-daily").config('spark.driver.extraJavaOptions', '-Duser.timezone=UTC').config('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')\
        .config("spark.sql.datetime.java8API.enabled", "true").config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .getOrCreate()


# In[9]:


# load the APC data from a prepared file
processed_triplevel = os.path.join('../data', 'processed', 'triplevel_df.parquet')
if not os.path.exists(processed_triplevel):
# if True:
    filepath = os.path.join(os.getcwd(), "../data", "processed", "apc_weather_gtfs_20220921.parquet")
    apcdata = spark.read.load(filepath)
    apcdata.createOrReplaceTempView("apc")

    # filter subset
    query = f"""
                SELECT *
                FROM apc
            """
    apcdata=spark.sql(query)
    apcdata = data_utils.remove_nulls_from_apc(apcdata)
    apcdata.createOrReplaceTempView('apcdata')
    apcdata_per_trip = data_utils.get_apc_per_trip_sparkview(spark)
    df = apcdata_per_trip.toPandas()
    
    # Adding extra features
    # Holidays
    fp = os.path.join('../data', 'others', 'US Holiday Dates (2004-2021).csv')
    holidays_df = pd.read_csv(fp)
    holidays_df['Date'] = pd.to_datetime(holidays_df['Date'])
    holidays_df['is_holiday'] = True
    df = df.merge(holidays_df[['Date', 'is_holiday']], left_on='transit_date', right_on='Date', how='left')
    df['is_holiday'] = df['is_holiday'].fillna(False)
    df = df.drop(columns=['Date'])
    
    # School breaks
    fp = os.path.join('../data', 'others', 'School Breaks (2019-2022).pkl')
    school_break_df = pd.read_pickle(fp)
    school_break_df['is_school_break'] = True
    df = df.merge(school_break_df[['Date', 'is_school_break']], left_on='transit_date', right_on='Date', how='left')
    df['is_school_break'] = df['is_school_break'].fillna(False)
    df = df.drop(columns=['Date'])

    df.to_parquet(processed_triplevel, engine='auto', compression='gzip')
else:
    df = pd.read_parquet(processed_triplevel, engine='auto')
    df = df.dropna()
    # Removing time_window in case a different one will be used
df = df.drop(['time_window', 'load'], axis=1)
df = df.reset_index(drop=True)


# ## Feature Analysis (used features)
# * Datetime: `year`, `month`, `dayofweek`, `hour`, `day`
# * GTFS: `scheduled_headway`, `route_direction_name`, `route_id`, `block_abbr`
# * Weather: `temperature`, `humidity`, `precipitation_intensity`
# * APC data on a stop level is grouped into trips and data is gathered by using the first instance (route_id, route_direction_name) or the average of the numerical values (scheduled headay, weather data)

# In[10]:


print(df.shape)
df.head(1)


# In[11]:


df.transit_date.max()


# ## Feature Generation
# Generated features, $y_t = f(x_{t-1})$, are always generated using past information.
# * `time_window`: Assigning the arrival times into time windows (30 minutes by default).
# * `window_of_day`: Just a larger time window (could probably remove)
# * `actual_headways`: On a stop level, actual headways are given using the arrival times of the bus to the bus stop. On a trip level, this was averaged over the multiple bus stops across a single trip.
# * `congestion_surrogate`: Generated by a model trained on the scheduled and actual headways. (tentatively included, surrogate model is not yet that accurate)
# * `route_id_direction`: Combined route_id and route_direction into one feature and then one hot encoded.
# * Other categorical values are converted to ordinal integers.

# In[12]:


RANDOM_SEED = 100
WINDOW = 30
PAST_TRIPS = 5
TARGET = 'y_reg100'


# In[13]:


cat_features = ['route_id_direction', 'is_holiday', 'dayofweek', 'is_school_break', 'time_window']
ord_features = ['year', 'month', 'hour', 'day']
# num_features = ['temperature', 'humidity', 'precipitation_intensity', 'avg_sched_headway', 'traffic_speed']
num_features = ['temperature', 'humidity', 'precipitation_intensity', 'avg_sched_headway']


# In[14]:


df = df.query("transit_date <= '2022-06-30'")
df


# In[15]:


# In the interest of time
tdf = deepcopy(df)


# In[16]:


tdf = triplevel_utils.generate_new_features(df, time_window=WINDOW, past_trips=PAST_TRIPS, target=TARGET)


# In[17]:


# Group by time windows and get the maximum of the aggregate load/class/sched
# Get mean of temperature (mostly going to be equal)
tdf = tdf.groupby(['transit_date', 'route_id_direction', 'time_window']).agg({"year":"first", 
                                                                              "month":"first",
                                                                              "day": "first",
                                                                              "dayofweek":"first", 
                                                                              "hour":"first",
                                                                              "is_holiday": "first",
                                                                              "is_school_break": "first",
                                                                              "temperature":"mean", 
                                                                              "humidity":"mean",
                                                                              "precipitation_intensity": "mean",
                                                                              "scheduled_headway": "max",
                                                                              TARGET: "max"})
                                                                            #   "traffic_speed":"mean",
tdf = tdf.reset_index(level=[0,1,2])



print("ohe_encoder is for the following column order:", cat_features)
rf_df, ix_map, ohe_encoder, percentiles = triplevel_utils.prepare_df_for_training(tdf, cat_features, ord_features, target=TARGET)
percentiles


# In[20]:


drop_cols = ['route_id', 'route_direction_name', 'block_abbr', 'y_reg100', 'y_reg095', 'transit_date', 'is_holiday', 'route_id_direction', 'actual_headways', 'trip_id', 'arrival_time']
drop_cols = [col for col in drop_cols if col in rf_df.columns]
rf_df = rf_df.drop(drop_cols, axis=1)

y = rf_df.pop('y_class')
X = rf_df


# In[21]:


print(X.shape)
X.head(5).style.set_precision(2)


# In[22]:


print(y.unique())
pd.DataFrame(y.head())


# In[23]:


fp = os.path.join('../models', 'any_day', 'TL_OHE_encoders.joblib')
joblib.dump(ohe_encoder, fp)
fp = os.path.join('../models', 'any_day', 'TL_IX_map.joblib')
joblib.dump(ix_map, fp)
fp = os.path.join('../models', 'any_day', 'TL_X_columns.joblib')
joblib.dump(X.columns, fp)


# In[23]:


# Grid search results
fp = os.path.join('../models', 'any_day', 'XGBOOST_RANDSEARCHCV_any_day_with_schoolbreak012.joblib')
search_results = joblib.load(fp)
print(search_results.best_params_)


# # For bins 012

# In[24]:


# Train on entire dataset

n_estimators  = search_results.best_params_['n_estimators']
max_depth     = search_results.best_params_['max_depth']
learning_rate = search_results.best_params_['learning_rate']
gamma         = search_results.best_params_['gamma']
objective     = 'multi:softmax'

model012 = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                             learning_rate=learning_rate, use_label_encoder=False, gamma=gamma, num_class=3,
                             objective=objective, eval_metric='mlogloss')
# model012 = xgb.XGBClassifier(use_label_encoder=False, num_class=3,
#                              objective=objective, eval_metric='mlogloss')

model012.fit(X, y, verbose=1)

fp = os.path.join('../models', 'any_day', 'XGB_012.joblib')
joblib.dump(model012, fp)


# ## For bins 234

# In[25]:


rf_df, ix_map, ohe_encoder, percentiles = triplevel_utils.prepare_df_for_training(tdf, cat_features, ord_features, target=TARGET)

# Train 2 separate models for bins 0, 1, 2 and 2, 3, 4
# Adjusting y_class to incorporate Dan's request
# Use Transit's 3 bins as a base. For the highest capacity bin, carve out everything from 55 to 75 as a 4th bin, and 75+ as a 5th bin.

rf_df, percentiles = triplevel_utils.adjust_bins(rf_df, TARGET=TARGET, percentiles=percentiles)
# display(rf_df['y_class'].value_counts())
print(percentiles)
drop_cols = ['route_id', 'route_direction_name', 'block_abbr', 'y_reg100', 'y_reg095', 'transit_date', 'is_holiday', 'route_id_direction', 'is_school_break']
drop_cols = [col for col in drop_cols if col in rf_df.columns]
rf_df = rf_df.drop(drop_cols, axis=1)
rf_df = rf_df[rf_df['y_class'] >= 2]
# display(rf_df['y_class'].value_counts())

y = rf_df.pop('y_class')
y = y - 2
X = rf_df


# In[ ]:


# Grid search results
fp = os.path.join('../models', 'any_day', 'XGBOOST_RANDSEARCHCV_any_day_with_schoolbreak234.joblib')
search_results = joblib.load(fp)
print(search_results.best_params_)


# In[26]:


# Train on entire dataset
n_estimators  = search_results.best_params_['n_estimators']
max_depth     = search_results.best_params_['max_depth']
learning_rate = search_results.best_params_['learning_rate']
gamma         = search_results.best_params_['gamma']
objective     = 'multi:softmax'

model234 = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                            learning_rate=learning_rate, use_label_encoder=False, gamma=gamma, num_class=3,
                            objective=objective, eval_metric='mlogloss')
# model234 = xgb.XGBClassifier(use_label_encoder=False, num_class=3,
#                              objective=objective, eval_metric='mlogloss')

model234.fit(X, y, verbose=1)

fp = os.path.join('../models', 'any_day', 'XGB_234.joblib')
joblib.dump(model234, fp)


# ### Prediction for testing

# In[161]:


# load the APC data from a prepared file
processed_triplevel = os.path.join('../data', 'processed', 'triplevel_df.parquet')
df = pd.read_parquet(processed_triplevel, engine='auto')
df = df.dropna()
df = df.drop(['time_window', 'load'], axis=1)
df = df.reset_index(drop=True)
df = df.query("transit_date > '2022-06-30'")

tdf = triplevel_utils.generate_new_features(df, time_window=WINDOW, past_trips=PAST_TRIPS, target=TARGET)
# Group by time windows and get the maximum of the aggregate load/class/sched
# Get mean of temperature (mostly going to be equal)
tdf = tdf.groupby(['transit_date', 'route_id_direction', 'time_window']).agg({"year":"first", 
                                                                              "month":"first",
                                                                              "day": "first",
                                                                              "dayofweek":"first", 
                                                                              "hour":"first",
                                                                              "is_holiday": "first",
                                                                              "is_school_break": "first",
                                                                              "temperature":"mean", 
                                                                              "humidity":"mean",
                                                                              "precipitation_intensity": "mean",
                                                                              "scheduled_headway": "max",
                                                                              TARGET: "max"})
tdf = tdf.reset_index(level=[0,1,2])
# display(tdf.head())
print("ohe_encoder is for the following column order:", cat_features)


# In[162]:


# OHE for route_id_direction
columns     = joblib.load('../models/any_day/TL_X_columns.joblib')
ix_map      = joblib.load('../models/any_day/TL_IX_map.joblib')
ohe_encoder = joblib.load('../models/any_day/TL_OHE_encoders.joblib')

transit_dates = tdf.pop('transit_date')
y = tdf.pop('y_reg100')
tdf = tdf

tdf[ohe_encoder.get_feature_names_out()] = ohe_encoder.transform(tdf[cat_features]).toarray()
tdf = tdf.drop(columns=cat_features)

# label encode of categorical variables
for col in ord_features:
    tdf[f'{col}_ix'] = tdf[col].apply(lambda x: ix_map[col][x])
tdf = tdf.drop(columns=ord_features)

tdf = tdf[columns]
tdf = tdf.dropna()
tdf[columns[1:6]] = tdf[columns[1:6]].apply(pd.to_numeric)
rf_df = deepcopy(tdf)


# In[163]:


drop_cols = ['route_id', 'route_direction_name', 'block_abbr', 'y_reg100', 'y_reg095', 'transit_date', 'is_holiday', 'route_id_direction', 'actual_headways', 'trip_id', 'arrival_time']
drop_cols = [col for col in drop_cols if col in rf_df.columns]
rf_df = rf_df.drop(drop_cols, axis=1)
X = rf_df


# In[164]:


model012    = joblib.load('../models/any_day/XGB_012.joblib')
model234    = joblib.load('../models/any_day/XGB_234.joblib')


# In[165]:


## Predict first stage 0-1-2
predictions = model012.predict(X)

unique, counts = np.unique(predictions, return_counts=True)
print(unique, counts)

X['y_pred'] = predictions
## Isolate predictions with bin 2 for 2-3-4
high_bin_df = X[X['y_pred'] == 2]

high_bin_df = high_bin_df.drop(['y_pred'], axis=1)
high_bin_index = high_bin_df.index
high_bin_df = high_bin_df[columns]

predictions = model234.predict(high_bin_df)

unique, counts = np.unique(predictions, return_counts=True)
print(unique, counts)

predictions = predictions + 2
X.loc[high_bin_index, 'y_pred'] = predictions
X[cat_features] = ohe_encoder.inverse_transform(X.filter(regex='route_id_direction_|is_holiday_|dayofweek_|is_school_break_|time_window_'))

X = X[X.columns.drop(list(X.filter(regex='route_id_direction_|is_holiday_|dayofweek_|is_school_break_|time_window_')))]

for col in ord_features:
    inv_map = {v: k for k, v in ix_map[col].items()}
    X[col] = X[f'{col}_ix'].apply(lambda x: inv_map[x])
X = X.drop(columns=[f"{col}_ix" for col in ord_features], axis=1)
X['load'] = y
X['transit_date'] = transit_dates

labels = [(0, 9), (10, 16), (17, 55), (56, 75), (76, 100)]
percentiles = [(-1, 9), (9, 16), (16, 55), (55, 75), (75, 100)]
bins = pd.IntervalIndex.from_tuples(percentiles)
mycut = pd.cut(X['load'].tolist(), bins=bins)
X['y_class'] = mycut.codes

rearrange = ['transit_date','temperature', 'humidity', 'precipitation_intensity',
            'scheduled_headway', 'route_id_direction', 'is_holiday', 'is_school_break',
            'dayofweek', 'time_window', 'year', 'month', 'hour', 'day', 'load', 'y_class', 'y_pred']
X = X[rearrange]


# In[166]:


fp = '/home/jptalusan/mta_stationing_problem/models/any_day/20220701_to_20220919_any_day_results_GRID.pkl'
X.to_pickle(fp)