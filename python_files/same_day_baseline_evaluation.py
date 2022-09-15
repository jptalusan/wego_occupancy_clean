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
from multiprocessing import Process, Queue, cpu_count, Manager
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

# Just gets max of past rows
def generate_simple_baseline_predictions(input_df, past, future):
    prediction = []
    past_df = input_df[:past]
    future_df = input_df[past:]
    for f in range(future):
        y_pred = past_df['y_class'].max()
        prediction.append(y_pred)
        last_row = future_df.iloc[[0]]
        past_df = pd.concat([past_df[1:past], last_row])
        future_df = future_df.iloc[1: , :]
    return prediction

# Get max of past loads from the same stop sequence/trip/
def generate_stop_level_baseline(trip_id, transit_date, route_id_dir, future, lookback=10):
    trip_id = trip_id
    transit_date = transit_date
    route_id_dir = route_id_dir
    
    _df = df[(df['trip_id'] == trip_id) & (df['transit_date'] == transit_date)]
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

f = os.path.join('../data', 'processed', 'apc_weather_gtfs.parquet')
apcdata = spark.read.load(f)
todelete = apcdata.filter('(load < 0) OR (load IS NULL)').select('transit_date','trip_id','overload_id').distinct()
todelete=todelete.withColumn('marker',F.lit(1))

test_dates =  ('2021-09-30', '2022-04-06')

#joining and whereever the records are not found in sync error table the marker will be null
apcdataafternegdelete = apcdata.join(todelete,on=['trip_id','transit_date','overload_id'],how='left').filter('marker is null').drop('marker')
apcdataafternegdelete = apcdataafternegdelete.sort(['trip_id', 'overload_id'])

get_columns = ['trip_id', 'transit_date', 'arrival_time', 
               'block_abbr', 'stop_sequence', 'stop_id_original',
               'load', 
               'route_direction_name', 'route_id',
               'dayofweek',  'year', 'month', 'hour']
get_str = ", ".join([c for c in get_columns])

apcdataafternegdelete.createOrReplaceTempView("apc")

# # filter subset
query = f"""
SELECT {get_str}
FROM apc
WHERE transit_date >= '{test_dates[0]}' AND transit_date <= '{test_dates[1]}'
"""

apcdataafternegdelete = spark.sql(query)
df = apcdataafternegdelete.toPandas()
df = df[df.arrival_time.notna()]

df['route_id_dir'] = df["route_id"].astype("str") + "_" + df["route_direction_name"]

df = df.sort_values(['transit_date'])
df['day'] = df["arrival_time"].dt.day

TIMEWINDOW = 15
df['minute'] = df['arrival_time'].dt.minute
df['minuteByWindow'] = df['minute'] // TIMEWINDOW
df['temp'] = df['minuteByWindow'] + (df['hour'] * 60 / TIMEWINDOW)
df['time_window'] = np.floor(df['temp']).astype('int')
df = df.drop(columns=['minute', 'minuteByWindow', 'temp'])

df = df.groupby(['transit_date', 
                'route_id_dir', 
                'stop_id_original',
                'time_window']).agg({"trip_id":"first",
                                    "block_abbr":"first",
                                    "arrival_time":"first",
                                    "year":"first", 
                                    "month":"first",
                                    "day": "first",
                                    "hour":"first",
                                    "dayofweek":"first",
                                    "stop_sequence":"first",
                                    "load": "sum" })
                
df = df.reset_index(level=[0,1,2,3])
df = df.sort_values(by=['block_abbr', 'arrival_time']).reset_index(drop=True)

percentiles = [(0.0, 6.0), (7.0, 12.0), (13.0, 55.0), (56.0, 75.0), (76.0, 100.0)]
df['y_class'] = df['load'].apply(lambda x: data_utils.get_class(x, percentiles))

fp = os.path.join('../models/same_day/evaluation/random_trip_df_5000.pkl')
random_trip_df = pd.read_pickle(fp)
random_trip_df = random_trip_df

PAST = 5
def process_baseline_maxmean(L, queue):
    while queue.qsize() > 0 :
        random_trip = queue.get()
        trip_df = df.merge(random_trip, on=['transit_date', 'trip_id', 'route_id_dir']).sort_values(['stop_sequence'])
        if trip_df.empty:
            return pd.DataFrame()
        future = len(trip_df) - PAST
        y_true = trip_df.iloc[PAST:].y_class.tolist()
        
        trip_id = trip_df.iloc[0]['trip_id']
        transit_date = trip_df.iloc[0]['transit_date']
        route_id_dir = trip_df.iloc[0]['route_id_dir']
        trip_df['y_pred_max'] = -1
        trip_df['y_pred_mean'] = -1
        trip_df['y_pred_roll'] = -1
        
        y_pred_max, y_pred_ave = generate_stop_level_baseline(trip_id, transit_date, route_id_dir, future)
        y_pred_roll = generate_simple_baseline_predictions(trip_df, PAST, future)
        
        trip_df.loc[trip_df.iloc[PAST:].index, 'y_pred_max'] = y_pred_max
        trip_df.loc[trip_df.iloc[PAST:].index, 'y_pred_mean'] = y_pred_ave
        trip_df.loc[trip_df.iloc[PAST:].index, 'y_pred_roll'] = y_pred_roll
        
        L.append(trip_df)

queue = Queue()

def mp_handler():
    with Manager() as manager:
        L = manager.list()
        
        # Spawn two processes, assigning the method to be executed 
        # and the input arguments (the queue)
        processes = [Process(target=process_baseline_maxmean, args=(L,queue,)) for _ in range(cpu_count() - 1)]

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        res_df = pd.concat(L)

        fp = os.path.join('../evaluation/same_day/clean_evaluation/timewindow15', f'baseline_5000.pkl')
        res_df.to_pickle(fp)

if __name__ == '__main__':

    for k, v in random_trip_df.iterrows():
        queue.put(v.to_frame().T)

    mp_handler()