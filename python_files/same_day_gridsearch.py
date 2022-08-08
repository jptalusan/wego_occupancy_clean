from tensorflow.keras import backend as K
K.clear_session()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# os.chdir("/media/seconddrive/mta_stationing_problem")
import sys
sys.path.insert(0,'..')
import sys
import datetime as dt
import importlib
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import SparkConf
import numpy as np
import pickle
import joblib
import matplotlib as mpl
import tensorflow as tf
from tensorflow import keras
from tqdm import trange, tqdm

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
importlib.reload(tf_utils)
importlib.reload(models)

spark = SparkSession.builder.config('spark.executor.cores', '8').config('spark.executor.memory', '80g')\
        .config("spark.sql.session.timeZone", "UTC").config('spark.driver.memory', '40g').master("local[26]")\
        .appName("wego-daily").config('spark.driver.extraJavaOptions', '-Duser.timezone=UTC').config('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')\
        .config("spark.sql.datetime.java8API.enabled", "true").config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .config("spark.sql.autoBroadcastJoinThreshold", -1)\
        .config("spark.driver.maxResultSize", 0)\
        .config("spark.shuffle.spill", "true")\
        .getOrCreate()

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
df = apcdataafternegdelete.toPandas()
print(df.shape)
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

# Traffic
# Causes 3M data points to be lost
fp = os.path.join('../data', 'traffic', 'triplevel_speed.pickle')
speed_df = pd.read_pickle(fp)
speed_df = speed_df.rename({'route_id_direction':'route_id_dir'}, axis=1)
speed_df = speed_df[['transit_date', 'trip_id', 'route_id_dir', 'traffic_speed']]
df = df.merge(speed_df, how='left', 
                left_on=['transit_date', 'trip_id', 'route_id_dir'], 
                right_on=['transit_date', 'trip_id', 'route_id_dir'])
df['traffic_speed'].bfill(inplace=True)

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

TIMEWINDOW = 15
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
                                                      "dayofweek":"first",
                                                      "stop_sequence":"first",
                                                      "darksky_temperature":"mean", 
                                                      "darksky_humidity":"mean",
                                                      "darksky_precipitation_probability": "mean",
                                                      "traffic_speed":"mean",
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

train_dates = ('2020-01-01', '2021-06-30')
val_dates =   ('2021-06-30', '2021-10-31')
test_dates =  ('2021-10-31', '2022-04-06')

target = 'y_class'

num_columns = ['darksky_temperature', 'darksky_humidity', 'darksky_precipitation_probability', 'sched_hdwy', 'traffic_speed']
cat_columns = ['month', 'hour', 'day', 'stop_sequence', 'stop_id_original', 'year', 'time_window', target]
ohe_columns = ['dayofweek', 'route_id_dir', 'is_holiday']

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

test_df['unique_trip'] = test_df['trip_id'] + '_' + test_df['transit_date'].dt.strftime('%Y-%m-%d')

drop_cols = ['transit_date', 'load', 'arrival_time', 'trip_id']
drop_cols = [col for col in drop_cols if col in train_df.columns]
# train_df = train_df.drop(drop_cols, axis=1)
# val_df = val_df.drop(drop_cols, axis=1)
test_df = test_df.drop(drop_cols, axis=1)

arrange_cols = [target] + [col for col in test_df.columns if col != target]
# train_df = train_df[arrange_cols]
# val_df = val_df[arrange_cols]
test_df = test_df[arrange_cols]

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

PAST = 5
fp = os.path.join('../models/same_day/evaluation/random_trip_ids_2000.pkl')
with open(fp, 'rb') as f:
    random_trip_ids = pickle.load(f)
    
# Load models
num_features = len(test_df.columns) - 1
simple_lstm = linklevel_utils.setup_simple_lstm_generator(num_features, len(test_df.y_class.unique()))
# Load model
latest = tf.train.latest_checkpoint('../models/same_day/model')

print(latest)
simple_lstm.load_weights(latest)

results = []
for unique_trip_id in tqdm(random_trip_ids):
    trip_df = test_df[test_df['unique_trip'] == unique_trip_id]
    drop_cols = ['trip_id', 'transit_date', 'unique_trip']
    drop_cols = [col for col in drop_cols if col in trip_df.columns]
    trip_df = trip_df.drop(drop_cols, axis=1)
    
    future = len(trip_df) - PAST
    past_df = trip_df.iloc[0:PAST]
    
    y_true = trip_df.iloc[PAST:].y_class.tolist()
    y_pred = generate_simple_lstm_predictions(past_df, simple_lstm, future)
    
    res_df = pd.DataFrame(np.column_stack(([unique_trip_id]*future, y_true, y_pred)), columns=['trip_id', 'y_true', 'y_pred'])
    results.append(res_df)
    
res_df = pd.concat(results)
fp = os.path.join('../models/same_day/evaluation', f'SIMPLE_LSTM_multi_stop_{PAST}P_xF_results.pkl')
res_df.to_pickle(fp)


# Get max of past loads from the same stop sequence/trip/
def generate_stop_level_baseline(unique_trip_id, past, future, lookback=10):
    trip_id = unique_trip_id.split("_")[0]
    transit_date = unique_trip_id.split("_")[1]
    
    _df = df[(df['trip_id'] == trip_id) & (df['transit_date'] == transit_date)]
    route_id_dir = _df['route_id_dir'].values[0]
    hour = _df['hour'].values[0]
    block_abbr = _df['block_abbr'].values[0]
    dayofweek = _df['dayofweek'].values[0]
    
    prediction_max = []
    prediction_ave = []
    for stop_sequence in range(future):
        load = df[(df['route_id_dir'] == route_id_dir) & 
                      (df['stop_sequence'] == stop_sequence) & 
                      (df['hour'] == hour) & 
                      (df['block_abbr'] == block_abbr) & 
                      (df['dayofweek'] == dayofweek)].sort_values(by='transit_date')[-lookback:]['load']
        max_load = load.max()
        ave_load = load.mean()
        y_pred = data_utils.get_class(max_load, percentiles)
        prediction_max.append(y_pred)
        y_pred = data_utils.get_class(ave_load, percentiles)
        prediction_ave.append(y_pred)
    return prediction_max, prediction_ave

PAST = 10
results = []
for unique_trip_id in tqdm(random_trip_ids):
    trip_df = test_df[test_df['unique_trip'] == unique_trip_id]
    if trip_df.empty:
        continue
    future = len(trip_df) - PAST
    y_true = trip_df.iloc[PAST:].y_class.tolist()
    y_pred_max, y_pred_ave = generate_stop_level_baseline(unique_trip_id, PAST, future)
    res_df = pd.DataFrame(np.column_stack(([unique_trip_id]*future, y_true, y_pred_max, y_pred_ave)), columns=['trip_id', 'y_true', 'y_pred_max', 'y_pred_ave'])
    results.append(res_df)

res_df = pd.concat(results)

fp = os.path.join('../models/same_day/evaluation', f'baseline_multi_stop_{PAST}P_xF_results.pkl')
res_df.to_pickle(fp)
