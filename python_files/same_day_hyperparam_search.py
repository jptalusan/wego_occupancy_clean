#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import datetime as dt
import importlib
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import SparkConf
import numpy as np
import pickle
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, concatenate, GlobalAveragePooling1D
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.models import Model
from tqdm import trange, tqdm
import sys
sys.path.insert(0,'..')
from src import tf_utils, config, data_utils, models, linklevel_utils
import logging
from itertools import product
import argparse
from tensorflow.keras import backend as K
K.clear_session()

import warnings
import pandas as pd
pd.set_option('display.max_columns', None)
from pandas.core.common import SettingWithCopyWarning

# Create the parser and add arguments
parser = argparse.ArgumentParser()
parser.add_argument('--past', dest='past', help="This is the first argument", type=int)
parser.add_argument('--timewindow', dest='timewindow', help="This is the first argument", type=int)

# Parse and print the results
args = parser.parse_args()
print("TW:", args.timewindow)
print("Past:", args.past)

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
tf.get_logger().setLevel('INFO')
importlib.reload(tf_utils)
importlib.reload(models)
from multiprocessing import Process, Queue, cpu_count, Manager

spark = SparkSession.builder.config('spark.executor.cores', '8').config('spark.executor.memory', '80g')\
    .config("spark.sql.session.timeZone", "UTC").config('spark.driver.memory', '40g').master("local[26]")\
    .appName("wego-daily").config('spark.driver.extraJavaOptions', '-Duser.timezone=UTC').config('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')\
    .config("spark.sql.datetime.java8API.enabled", "true").config("spark.sql.execution.arrow.pyspark.enabled", "true")\
    .config("spark.sql.autoBroadcastJoinThreshold", -1)\
    .config("spark.driver.maxResultSize", 0)\
    .config("spark.shuffle.spill", "true")\
    .getOrCreate()

OUTPUT_DIR = os.path.join('../models', 'same_day', 'gridsearch')

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
train_dates = ('2020-01-01', '2021-06-30')
val_dates =   ('2021-06-30', '2021-10-31')
test_dates =  ('2021-10-31', '2022-04-06')

target = 'y_class'

num_columns = ['darksky_temperature', 'darksky_humidity', 'darksky_precipitation_probability', 'sched_hdwy']
cat_columns = ['month', 'hour', 'day', 'stop_sequence', 'stop_id_original', 'year', target]
ohe_columns = ['dayofweek', 'route_id_dir', 'is_holiday', 'is_school_break', 'zero_load_at_trip_end', 'time_window']


# In[2]:


PAST = args.past
TIMEWINDOW = args.timewindow

# STATIC
past = PAST
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
hyperparams_dict


# In[3]:


OUTPUT_DIR = '/home/jptalusan/mta_stationing_problem/models/same_day/gridsearch'
CURR_RUN_DIR = f'TW_{TIMEWINDOW}_P_{PAST}'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, CURR_RUN_DIR)

RANDOM_TEST_TRIPS_PATH = '/home/jptalusan/mta_stationing_problem/models/same_day/evaluation/random_trip_df_5000.pkl'
HOLIDAYS_PATH = '/home/jptalusan/mta_stationing_problem/data/others/US Holiday Dates (2004-2021).csv'
SCHOOLBREAK_PATH = '/home/jptalusan/mta_stationing_problem/data/others/School Breaks (2019-2022).pkl'

Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)


# In[4]:


f = os.path.join('../data', 'processed', 'apc_weather_gtfs.parquet')
apcdata = spark.read.load(f)
todelete = apcdata.filter('(load < 0) OR (load IS NULL)').select('transit_date','trip_id','overload_id').distinct()
todelete=todelete.withColumn('marker',F.lit(1))

#joining and whereever the records are not found in sync error table the marker will be null
apcdataafternegdelete=apcdata.join(todelete,on=['trip_id','transit_date','overload_id'],how='left').filter('marker is null').drop('marker')
apcdataafternegdelete = apcdataafternegdelete.sort(['trip_id', 'overload_id'])

get_columns = ['trip_id', 'transit_date', 'arrival_time', 
                'block_abbr', 'stop_sequence', 'stop_id_original',
                'load', 
                'darksky_temperature', 
                'darksky_humidity', 
                'darksky_precipitation_probability', 
                'route_direction_name', 'route_id',
                'dayofweek',  'year', 'month', 'hour',
                'sched_hdwy', 'zero_load_at_trip_end']
get_str = ", ".join([c for c in get_columns])

apcdataafternegdelete.createOrReplaceTempView("apc")

# # filter subset
query = f"""
            SELECT {get_str}
            FROM apc
            """
apcdataafternegdelete = spark.sql(query)
apcdataafternegdelete = apcdataafternegdelete.na.fill(value=0,subset=["zero_load_at_trip_end"])
df = apcdataafternegdelete.toPandas()
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

sorted_df = []
for ba in tqdm(df.block_abbr.unique()):
    ba_df = df[df['block_abbr'] == ba]
    end_stop = ba_df.stop_sequence.max()
    # Same result as creating a fixed_arrival_time (but faster)
    ba_df = ba_df[ba_df.stop_sequence != end_stop].reset_index(drop=True)
    sorted_df.append(ba_df)
        
overall_df = pd.concat(sorted_df)
drop_cols = ['route_direction_name', 'route_id']
drop_cols = [col for col in drop_cols if col in overall_df.columns]
overall_df = overall_df.drop(drop_cols, axis=1)

overall_df['minute'] = overall_df['arrival_time'].dt.minute
overall_df['minuteByWindow'] = overall_df['minute'] // TIMEWINDOW
overall_df['temp'] = overall_df['minuteByWindow'] + (overall_df['hour'] * 60 / TIMEWINDOW)
overall_df['time_window'] = np.floor(overall_df['temp']).astype('int')
overall_df = overall_df.drop(columns=['minute', 'minuteByWindow', 'temp'])

## Aggregate stops by time window
# Group by time windows and get the maximum of the aggregate load/class/sched
# Get mean of temperature (mostly going to be equal)
# TODO: Double check this! 
overall_df = overall_df.groupby(['transit_date', 
                                'route_id_dir', 
                                'stop_id_original',
                                'time_window']).agg({"trip_id":"first",
                                                     "block_abbr":"first",
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
                                                     "sched_hdwy": "max",
                                                     "load": "sum" })
overall_df = overall_df.reset_index(level=[0,1,2,3])
overall_df = overall_df.sort_values(by=['block_abbr', 'arrival_time']).reset_index(drop=True)

drop_cols = ['arrival_time', 'block_abbr']
drop_cols = [col for col in drop_cols if col in overall_df.columns]
overall_df = overall_df.drop(drop_cols, axis=1)
# checking bins of loads for possible classification problem
loads = overall_df[overall_df.load <= config.TARGET_MAX]['load']
percentiles = []
for cbin in config.CLASS_BINS:
    percentile = np.percentile(loads.values, cbin)
    percentiles.append(percentile)

percentiles = [(percentiles[0], percentiles[1]), (percentiles[1] + 1, percentiles[2]), (percentiles[2] + 1, 55.0), (56.0, 75.0), (76.0, 100.0)]
    
print(f"Percentiles: {percentiles}")
overall_df[config.TARGET_COLUMN_CLASSIFICATION] = overall_df['load'].apply(lambda x: data_utils.get_class(x, percentiles))
overall_df = overall_df[overall_df[config.TARGET_COLUMN_CLASSIFICATION].notna()]

ohe_encoder, label_encoder, num_scaler, train_df, val_df, test_df = linklevel_utils.prepare_linklevel(overall_df, 
                                                                                                        train_dates=train_dates, 
                                                                                                        val_dates=val_dates, 
                                                                                                        test_dates=test_dates,
                                                                                                        cat_columns=cat_columns,
                                                                                                        num_columns=num_columns,
                                                                                                        ohe_columns=ohe_columns,
                                                                                                        feature_label='y_class',
                                                                                                        time_feature_used='transit_date',
                                                                                                        scaler='minmax')

drop_cols = ['transit_date', 'load', 'trip_id', 'arrival_time'] + ohe_columns
drop_cols = [col for col in drop_cols if col in train_df.columns]
train_df = train_df.drop(drop_cols, axis=1)
val_df = val_df.drop(drop_cols, axis=1)

arrange_cols = [target] + [col for col in train_df.columns if col != target]
train_df = train_df[arrange_cols]
val_df = val_df[arrange_cols]

train_df['y_class'] = train_df.y_class.astype('int')
val_df['y_class']   = val_df.y_class.astype('int')
test_df['y_class']  = test_df.y_class.astype('int')


# In[5]:


## Saving encoders, scalers and column arrangement
fp = os.path.join(OUTPUT_PATH, 'LL_OHE_encoder.joblib')
joblib.dump(ohe_encoder, fp)
fp = os.path.join(OUTPUT_PATH, 'LL_Label_encoders.joblib')
joblib.dump(label_encoder, fp)
fp = os.path.join(OUTPUT_PATH, 'LL_Num_scaler.joblib')
joblib.dump(num_scaler, fp)
fp = os.path.join(OUTPUT_PATH, 'LL_X_columns.joblib')
joblib.dump(train_df.columns, fp)

print(f"Done saving joblibs")

# In[7]:


drop_cols = ['transit_date', 'load', 'trip_id', 'arrival_time'] + ohe_columns
drop_cols = [col for col in drop_cols if col in train_df.columns]
train_df = train_df.drop(drop_cols, axis=1)
val_df = val_df.drop(drop_cols, axis=1)


# In[9]:


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

label_index = train_df.columns.tolist().index(target)
print(label_index)
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

# dataset_val = timeseries_dataset_from_dataset(val_df, 
#                                               feature_slice=feature_slice,
#                                               label_slice=label_slice,
#                                               input_sequence_length=input_sequence_length, 
#                                               output_sequence_length=output_sequence_length, 
#                                               batch_size=batch_size)

# dataset_test = timeseries_dataset_from_dataset(test_df,
#                                                feature_slice=feature_slice,
#                                                label_slice=label_slice,
#                                                input_sequence_length=input_sequence_length, 
#                                                output_sequence_length=output_sequence_length, 
#                                                batch_size=batch_size)
    


# In[11]:


num_classes = len(train_df.y_class.unique())
num_classes
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

from tensorflow.keras import backend as K
K.clear_session()

checkpoint_filepath = os.path.join(OUTPUT_PATH, 'CLA_cp-epoch{epoch:02d}-loss{loss:.2f}.ckpt')
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)
# fit model


print(f"Start training...")
callbacks = [keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True), model_checkpoint_callback]
history = model.fit(dataset_train, epochs=1, callbacks=callbacks, verbose=1)


# ## TESTING

# In[37]:


f = os.path.join('../data', 'processed', 'apc_weather_gtfs.parquet')
apcdata = spark.read.load(f)
todelete = apcdata.filter('(load < 0) OR (load IS NULL)').select('transit_date','trip_id','overload_id').distinct()
todelete=todelete.withColumn('marker',F.lit(1))

#joining and whereever the records are not found in sync error table the marker will be null
apcdataafternegdelete=apcdata.join(todelete,on=['trip_id','transit_date','overload_id'],how='left').filter('marker is null').drop('marker')
apcdataafternegdelete = apcdataafternegdelete.sort(['trip_id', 'overload_id'])

get_columns = ['trip_id', 'transit_date', 'arrival_time', 
                'block_abbr', 'stop_sequence', 'stop_id_original',
                'load', 
                'darksky_temperature', 
                'darksky_humidity', 
                'darksky_precipitation_probability', 
                'route_direction_name', 'route_id',
                'dayofweek',  'year', 'month', 'hour',
                'sched_hdwy', 'zero_load_at_trip_end']
get_str = ", ".join([c for c in get_columns])

apcdataafternegdelete.createOrReplaceTempView("apc")

# # filter subset
query = f"""
SELECT {get_str}
FROM apc
WHERE transit_date >= '{test_dates[0]}' AND transit_date <= '{test_dates[1]}'
"""

apcdataafternegdelete = spark.sql(query)
apcdataafternegdelete = apcdataafternegdelete.na.fill(value=0,subset=["zero_load_at_trip_end"])
df = apcdataafternegdelete.toPandas()
df = df[df.arrival_time.notna()]
df = df[df.sched_hdwy.notna()]
df = df[df.darksky_temperature.notna()]

df['route_id_dir'] = df["route_id"].astype("str") + "_" + df["route_direction_name"]
df['day'] = df["arrival_time"].dt.day
df = df.sort_values(by=['block_abbr', 'arrival_time']).reset_index(drop=True)

# Adding extra features
# Holidays
fp = os.path.join(HOLIDAYS_PATH)
holidays_df = pd.read_csv(fp)
holidays_df['Date'] = pd.to_datetime(holidays_df['Date'])
holidays_df['is_holiday'] = True
df = df.merge(holidays_df[['Date', 'is_holiday']], left_on='transit_date', right_on='Date', how='left')
df['is_holiday'] = df['is_holiday'].fillna(False)
df = df.drop(columns=['Date'])
    
# School breaks
fp = os.path.join(SCHOOLBREAK_PATH)
school_break_df = pd.read_pickle(fp)
school_break_df['is_school_break'] = True
df = df.merge(school_break_df[['Date', 'is_school_break']], left_on='transit_date', right_on='Date', how='left')
df['is_school_break'] = df['is_school_break'].fillna(False)
df = df.drop(columns=['Date'])

sorted_df = []
for ba in tqdm(df.block_abbr.unique()):
    ba_df = df[df['block_abbr'] == ba]
    end_stop = ba_df.stop_sequence.max()
    # Same result as creating a fixed_arrival_time (but faster)d
    ba_df = ba_df[ba_df.stop_sequence != end_stop].reset_index(drop=True)
    sorted_df.append(ba_df)
        
overall_df = pd.concat(sorted_df)
drop_cols = ['route_direction_name', 'route_id']
drop_cols = [col for col in drop_cols if col in overall_df.columns]
overall_df = overall_df.drop(drop_cols, axis=1)

overall_df['minute'] = overall_df['arrival_time'].dt.minute
overall_df['minuteByWindow'] = overall_df['minute'] // TIMEWINDOW
overall_df['temp'] = overall_df['minuteByWindow'] + (overall_df['hour'] * 60 / TIMEWINDOW)
overall_df['time_window'] = np.floor(overall_df['temp']).astype('int')
overall_df = overall_df.drop(columns=['minute', 'minuteByWindow', 'temp'])
overall_df = overall_df.sort_values(by=['block_abbr', 'arrival_time']).reset_index(drop=True)


# In[38]:


def prepare_test_linklevel(test_df, ohe_encoder, num_scaler, label_encoders,
                        cat_columns=None, num_columns=None, ohe_columns=None, feature_label='load'):
    test_df[ohe_encoder.get_feature_names_out()] = ohe_encoder.transform(test_df[ohe_columns]).toarray()
    for col in [col for col in cat_columns if col != feature_label]:
        encoder = label_encoders[col]
        test_df[col] = encoder.transform(test_df[col])
    test_df[num_columns] = num_scaler.transform(test_df[num_columns])
    return test_df


# In[39]:


from copy import deepcopy
test_df = deepcopy(overall_df)

columns = joblib.load(f'{OUTPUT_PATH}/LL_X_columns.joblib')
label_encoders = joblib.load(f'{OUTPUT_PATH}/LL_Label_encoders.joblib')
ohe_encoder = joblib.load(f'{OUTPUT_PATH}/LL_OHE_encoder.joblib')
num_scaler = joblib.load(f'{OUTPUT_PATH}/LL_Num_scaler.joblib')

test_df = prepare_test_linklevel(test_df, 
                                ohe_encoder, num_scaler, label_encoders,
                                cat_columns=cat_columns,
                                num_columns=num_columns,
                                ohe_columns=ohe_columns,
                                feature_label='y_class')


# In[40]:


def revere_transform(df, label_encoders, ohe_encoder):
    
    for col in cat_columns:
        if col == 'y_class':
            continue
        df[col] = label_encoders[col].inverse_transform(df[col])
        
    df[ohe_columns] = ohe_encoder.inverse_transform(df.filter(regex='dayofweek_|route_id_dir_|is_holiday_|is_school_break_|zero_load_at_trip_end_|time_window_'))
    df = df.drop(columns=df.filter(regex='dayofweek_|route_id_dir_|is_holiday_|is_school_|zero_load_|time_window_').columns, axis=1)
    return df


# In[41]:


def generate_simple_lstm_predictions(input_df, model, future):
    predictions = []
    for f in range(future):
        pred = model.predict(input_df.to_numpy().reshape(1, *input_df.shape))
        y_pred = np.argmax(pred)
        predictions.append(y_pred)
        last_row = input_df.iloc[[-1]]
        last_row['y_class'] = y_pred
        last_row['stop_sequence'] = last_row['stop_sequence'] + 1
        input_df = pd.concat([input_df[1:], last_row])
    return predictions


# In[42]:


print(f"Start testing...")
test_df['y_class'] = test_df['load'].apply(lambda x: data_utils.get_class(x, percentiles))
test_df = test_df[test_df['y_class'].notna()]


# In[45]:

# Load models
num_features = len(train_df.columns)
simple_lstm = linklevel_utils.setup_simple_lstm_generator(num_features, len(test_df.y_class.unique()))
# Load model
latest = tf.train.latest_checkpoint(OUTPUT_PATH)

simple_lstm.load_weights(latest)

# Load random trips for evaluation
fp = os.path.join(RANDOM_TEST_TRIPS_PATH)
random_trip_df = pd.read_pickle(fp)
results = []
for i in tqdm(range(len(random_trip_df[0:1000]))):
    trip_df = test_df.merge(random_trip_df.iloc[[i]], on=['transit_date', 'trip_id', 'route_id_dir'])
    
    inverse_trip_df = deepcopy(trip_df)
    inverse_trip_df = revere_transform(inverse_trip_df, label_encoders, ohe_encoder)
    drop_cols = ['transit_date', 'load', 'trip_id', 'arrival_time'] + ohe_columns
    drop_cols = [col for col in drop_cols if col in trip_df.columns]
    trip_df = trip_df.drop(drop_cols, axis=1)

    if len(trip_df) == 0:
        continue
    trip_df = trip_df[train_df.columns]
    
    future = len(trip_df) - PAST
    past_df = trip_df.iloc[0:PAST]
    
    inverse_trip_df['y_pred'] = -1

    y_true = trip_df.iloc[PAST:].y_class.tolist()
    y_pred = generate_simple_lstm_predictions(past_df, simple_lstm, future)
    inverse_trip_df.loc[trip_df.iloc[PAST:].index, 'y_pred'] = y_pred
    results.append(inverse_trip_df)

results = pd.concat(results)
fp = os.path.join(OUTPUT_PATH, 'results_1000_df.pkl')
results.to_pickle(fp)
