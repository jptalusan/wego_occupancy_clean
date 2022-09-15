#!/usr/bin/env python
# coding: utf-8

# In[38]:


from tensorflow.keras import backend as K
K.clear_session()


# In[39]:
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[40]:


import sys
import datetime as dt
import importlib
from pyspark import SparkContext,SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F
from pyspark import SparkConf
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, concatenate, GlobalAveragePooling1D
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import IPython
from copy import deepcopy
from tqdm import trange, tqdm

import sys
sys.path.insert(0,'..')
from src import tf_utils, config, data_utils, models, linklevel_utils

mpl.rcParams['figure.facecolor'] = 'white'

import warnings

import pandas as pd
import swifter
pd.set_option('display.max_columns', None)
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
tf.get_logger().setLevel('INFO')


# In[41]:


importlib.reload(tf_utils)
importlib.reload(models)


# In[42]:


import pyspark
print(pyspark.__version__)


# In[43]:


spark = SparkSession.builder.config('spark.executor.cores', '8').config('spark.executor.memory', '80g')\
        .config("spark.sql.session.timeZone", "UTC").config('spark.driver.memory', '40g').master("local[26]")\
        .appName("wego-daily").config('spark.driver.extraJavaOptions', '-Duser.timezone=UTC').config('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')\
        .config("spark.sql.datetime.java8API.enabled", "true").config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .config("spark.sql.autoBroadcastJoinThreshold", -1)\
        .config("spark.driver.maxResultSize", 0)\
        .config("spark.shuffle.spill", "true")\
        .getOrCreate()


# In[44]:


f = os.path.join('../data', 'processed', 'apc_weather_gtfs.parquet')
apcdata = spark.read.load(f)


# In[45]:


todelete = apcdata.filter('(load < 0) OR (load IS NULL)').select('transit_date','trip_id','overload_id').distinct()
todelete=todelete.withColumn('marker',F.lit(1))

#joining and whereever the records are not found in sync error table the marker will be null
apcdataafternegdelete=apcdata.join(todelete,on=['trip_id','transit_date','overload_id'],how='left').filter('marker is null').drop('marker')


# In[46]:


apcdataafternegdelete = apcdataafternegdelete.sort(['trip_id', 'overload_id'])


# In[47]:
# In[48]:


get_columns = ['trip_id', 'transit_date', 'arrival_time', 
               'block_abbr', 'stop_sequence', 'stop_id_original',
               'load', 
               'darksky_temperature', 
               'darksky_humidity', 
               'darksky_precipitation_probability', 
               'route_direction_name', 'route_id',
               'dayofweek',  'year', 'month', 'hour', 'zero_load_at_trip_end',
               'sched_hdwy']
get_str = ", ".join([c for c in get_columns])

apcdataafternegdelete.createOrReplaceTempView("apc")

# # filter subset
query = f"""
SELECT {get_str}
FROM apc
"""
print(query)

apcdataafternegdelete = spark.sql(query)
apcdataafternegdelete = apcdataafternegdelete.na.fill(value=0,subset=["zero_load_at_trip_end"])


# In[49]:


df = apcdataafternegdelete.toPandas()
old_shape = df.shape[0]


# In[50]:


df = df[df.arrival_time.notna()]
df = df[df.sched_hdwy.notna()]
df = df[df.darksky_temperature.notna()]

df['route_id_dir'] = df["route_id"].astype("str") + "_" + df["route_direction_name"]
df['day'] = df["arrival_time"].dt.day
df = df.sort_values(by=['block_abbr', 'arrival_time']).reset_index(drop=True)

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

# Traffic
# Causes 3M data points to be lost
fp = os.path.join('../data', 'traffic', 'triplevel_speed.pickle')
speed_df = pd.read_pickle(fp)
speed_df = speed_df.rename({'route_id_direction':'route_id_dir'}, axis=1)
speed_df = speed_df[['transit_date', 'trip_id', 'route_id_dir', 'traffic_speed']]
df = df.merge(speed_df, how='left', 
                left_on=['transit_date', 'trip_id', 'route_id_dir'], 
                right_on=['transit_date', 'trip_id', 'route_id_dir'])
# df = df[~df['traffic_speed'].isna()]
df['traffic_speed'].bfill(inplace=True)


# In[51]:


old_shape - df.shape[0]


# In[52]:


sorted_df = []
for ba in tqdm(df.block_abbr.unique()):
    ba_df = df[df['block_abbr'] == ba]
    end_stop = ba_df.stop_sequence.max()
    # Same result as creating a fixed_arrival_time (but faster)
    ba_df = ba_df[ba_df.stop_sequence != end_stop].reset_index(drop=True)
    sorted_df.append(ba_df)
        
overall_df = pd.concat(sorted_df)
drop_cols = ['route_direction_name', 'route_id', 'trip_id']
drop_cols = [col for col in drop_cols if col in overall_df.columns]
overall_df = overall_df.drop(drop_cols, axis=1)

# overall_df = overall_df.rename({"fixed_arrival_time": "arrival_time"}, axis=1)


# In[53]:


TIMEWINDOW = 15
overall_df['minute'] = overall_df['arrival_time'].dt.minute
overall_df['minuteByWindow'] = overall_df['minute'] // TIMEWINDOW
overall_df['temp'] = overall_df['minuteByWindow'] + (overall_df['hour'] * 60 / TIMEWINDOW)
overall_df['time_window'] = np.floor(overall_df['temp']).astype('int')
overall_df = overall_df.drop(columns=['minute', 'minuteByWindow', 'temp'])


# ## Aggregate stops by time window

# In[54]:



# In[55]:


# Group by time windows and get the maximum of the aggregate load/class/sched
# Get mean of temperature (mostly going to be equal)
# TODO: Double check this! 
overall_df = overall_df.groupby(['transit_date', 
                                 'route_id_dir', 
                                 'stop_id_original',
                                 'time_window']).agg({"block_abbr":"first",
                                                      "arrival_time":"first",
                                                      "year":"first", 
                                                      "month":"first",
                                                      "day": "first",
                                                      "hour":"first",
                                                      "is_holiday": "first",
                                                      "is_school_break":"first",
                                                      "dayofweek":"first",
                                                      "zero_load_at_trip_end":"first",
                                                      "stop_sequence":"first",
                                                      "darksky_temperature":"mean", 
                                                      "darksky_humidity":"mean",
                                                      "darksky_precipitation_probability": "mean",
                                                      "traffic_speed":"mean",
                                                      "sched_hdwy": "max",
                                                      "load": "sum" })
overall_df = overall_df.reset_index(level=[0,1,2,3])
overall_df = overall_df.sort_values(by=['block_abbr', 'arrival_time']).reset_index(drop=True)


# In[56]:


# In[57]:


drop_cols = ['arrival_time', 'block_abbr']
drop_cols = [col for col in drop_cols if col in overall_df.columns]
overall_df = overall_df.drop(drop_cols, axis=1)


# In[58]:


# checking bins of loads for possible classification problem
loads = overall_df[overall_df.load <= config.TARGET_MAX]['load']
percentiles = []
for cbin in config.CLASS_BINS:
    percentile = np.percentile(loads.values, cbin)
    percentiles.append(percentile)

# percentiles = [(percentiles[0], percentiles[1]), (percentiles[1] + 1, percentiles[2]), (percentiles[2] + 1, percentiles[3])]
percentiles = [(percentiles[0], percentiles[1]), (percentiles[1] + 1, percentiles[2]), (percentiles[2] + 1, 55.0), (56.0, 75.0), (76.0, 100.0)]
print(f"Percentiles: {percentiles}")
overall_df[config.TARGET_COLUMN_CLASSIFICATION] = overall_df['load'].apply(lambda x: data_utils.get_class(x, percentiles))
overall_df = overall_df[overall_df[config.TARGET_COLUMN_CLASSIFICATION].notna()]
overall_df.y_class.unique()


# In[59]:


# In[60]:


## Hyperparameters
past = 10 # Past stops observed
future = 1 # Future stops predicted
offset = 0

learning_rate = 1e-4
batch_size = 256
epochs = 200

feature_label = config.TARGET_COLUMN_CLASSIFICATION
patience = 10

hyperparams_dict = {'past': past,
                    'future': future,
                    'offset': offset,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'patience': patience}


# In[61]:


# target = config.TARGET_COLUMN_CLASSIFICATION
target = 'y_class'

num_columns = ['darksky_temperature', 'darksky_humidity', 'darksky_precipitation_probability', 'sched_hdwy', 'traffic_speed']
cat_columns = ['month', 'hour', 'day', 'stop_sequence', 'stop_id_original', 'year', target]
ohe_columns = ['dayofweek', 'is_holiday', 'is_school_break', 'zero_load_at_trip_end', 'time_window']

columns = num_columns + cat_columns + ohe_columns
print(f"Numerical columns: {num_columns}")
print(f"Categorical columns: {cat_columns}")
print(f"One Hot Encode columns: {ohe_columns}")


# In[62]:


overall_df.head(1)


# In[63]:


overall_df.hour.unique(), overall_df.stop_sequence.unique()


# In[ ]:


# Can add shuffle in the future
@tf.autograph.experimental.do_not_convert
def timeseries_dataset_from_dataset(df, feature_slice, label_slice, input_sequence_length, output_sequence_length, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(df.values)
    ds = dataset.window(input_sequence_length + output_sequence_length, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda x: x).batch(input_sequence_length + output_sequence_length)
     
    def split_feature_label(x):
        return x[:input_sequence_length:, feature_slice], x[input_sequence_length:,label_slice]
     
    ds = ds.map(split_feature_label)
     
    return ds.batch(batch_size)


# In[64]:


# Setup multiprocessing here
from pathlib import Path
from numba import cuda

route_id_dir_list = overall_df.route_id_dir.unique().tolist()
train_dates = ('2020-01-01', '2021-06-30')
val_dates =   ('2021-06-30', '2021-10-31')
test_dates =  ('2021-10-31', '2022-04-06')
for route_id_dir in route_id_dir_list[12:]:
    Path(f"../models/same_day/routes/{route_id_dir}").mkdir(parents=True, exist_ok=True)
    
    route_id_dir_df = overall_df.query("route_id_dir == @route_id_dir")
    route_id_dir_df = route_id_dir_df.drop(columns=['route_id_dir'])
    
    train_df = route_id_dir_df[(route_id_dir_df['transit_date'] >= train_dates[0]) &\
                  (route_id_dir_df['transit_date'] <= train_dates[1])]

    val_df = route_id_dir_df[(route_id_dir_df['transit_date'] >= val_dates[0]) &\
                (route_id_dir_df['transit_date'] <= val_dates[1])]

    test_df = route_id_dir_df[(route_id_dir_df['transit_date'] >= test_dates[0]) &\
                 (route_id_dir_df['transit_date'] <= test_dates[1])]
    
    if (len(train_df) == 0) or (len(val_df) == 0) or (len(test_df) == 0):
        continue

    ohe_encoder, label_encoder, num_scaler, train_df, val_df, test_df = linklevel_utils.prepare_linklevel(route_id_dir_df, 
                                                                                                            train_dates=train_dates, 
                                                                                                            val_dates=val_dates, 
                                                                                                            test_dates=test_dates,
                                                                                                            cat_columns=cat_columns,
                                                                                                            num_columns=num_columns,
                                                                                                            ohe_columns=ohe_columns,
                                                                                                            feature_label='y_class',
                                                                                                            time_feature_used='transit_date',
                                                                                                            scaler='minmax')

    drop_cols = ['transit_date', 'load', 'arrival_time'] + ohe_columns
    drop_cols = [col for col in drop_cols if col in train_df.columns]
    train_df = train_df.drop(drop_cols, axis=1)
    val_df = val_df.drop(drop_cols, axis=1)
    test_df = test_df.drop(drop_cols, axis=1)

    arrange_cols = [target] + [col for col in train_df.columns if col != target]
    train_df = train_df[arrange_cols]
    val_df = val_df[arrange_cols]
    test_df = test_df[arrange_cols]
    
    train_df['y_class'] = train_df.y_class.astype('int')
    val_df['y_class']   = val_df.y_class.astype('int')
    test_df['y_class']  = test_df.y_class.astype('int')
    
    ## Saving encoders, scalers and column arrangement
    fp = os.path.join('../models', 'same_day', 'routes', route_id_dir, 'LL_OHE_encoder.joblib')
    joblib.dump(ohe_encoder, fp)
    fp = os.path.join('../models', 'same_day', 'routes', route_id_dir, 'LL_Label_encoders.joblib')
    joblib.dump(label_encoder, fp)
    fp = os.path.join('../models', 'same_day', 'routes', route_id_dir, 'LL_Num_scaler.joblib')
    joblib.dump(num_scaler, fp)
    fp = os.path.join('../models', 'same_day', 'routes', route_id_dir, 'LL_X_columns.joblib')
    joblib.dump(train_df.columns, fp)
    
    label_index = train_df.columns.tolist().index(target)
    label_slice = slice(label_index, label_index + 1, None) # which column the label/labels are
    feature_slice = slice(None, None, None) # Which feature columns are included, by default includes all (even label)
    input_sequence_length = past # number of past information to look at
    output_sequence_length = future # number of time steps to predict
        
    dataset_train = timeseries_dataset_from_dataset(train_df, 
                                                    feature_slice=feature_slice,
                                                    label_slice=label_slice,
                                                    input_sequence_length=input_sequence_length, 
                                                    output_sequence_length=output_sequence_length, 
                                                    batch_size=batch_size)

    dataset_val = timeseries_dataset_from_dataset(val_df, 
                                                feature_slice=feature_slice,
                                                label_slice=label_slice,
                                                input_sequence_length=input_sequence_length, 
                                                output_sequence_length=output_sequence_length, 
                                                batch_size=batch_size)

    num_classes = len(train_df.y_class.unique())
    # define model
    model = tf.keras.Sequential()
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # compile model
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["sparse_categorical_accuracy"],
    )

    input_shape = (None, None, len(train_df.columns))
    model.build(input_shape)
    
    K.clear_session()
    
    model_filename = "CLA_cp-epoch{epoch:02d}-loss{val_loss:.2f}.ckpt"
    checkpoint_path = os.path.join('../models', 'same_day', 'routes', route_id_dir, model_filename)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    # fit model
    callbacks = [keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True), model_checkpoint_callback]

    history = model.fit(dataset_train, validation_data=dataset_val, epochs=5, callbacks=callbacks, verbose=1)


# In[ ]:




