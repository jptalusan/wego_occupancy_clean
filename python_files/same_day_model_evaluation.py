# Meant to work with whatever is in `same_day.ipynb` if for others, change accordingly.
from tensorflow.keras import backend as K
K.clear_session()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
sys.path.insert(0,'..')
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
import logging
import argparse
from copy import deepcopy
from pathlib import Path
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
importlib.reload(linklevel_utils)
importlib.reload(models)

spark = SparkSession.builder.config('spark.executor.cores', '8').config('spark.executor.memory', '80g')\
        .config("spark.sql.session.timeZone", "UTC").config('spark.driver.memory', '40g').master("local[26]")\
        .appName("wego-daily").config('spark.driver.extraJavaOptions', '-Duser.timezone=UTC').config('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')\
        .config("spark.sql.datetime.java8API.enabled", "true").config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .config("spark.sql.autoBroadcastJoinThreshold", -1)\
        .config("spark.driver.maxResultSize", 0)\
        .config("spark.shuffle.spill", "true")\
        .getOrCreate()

date_str = dt.datetime.today().strftime('%Y-%m-%d')
train_dates = ('2020-01-01', '2021-06-30')
val_dates =   ('2021-06-30', '2021-10-31')
test_dates =  ('2021-10-31', '2022-04-06')

target = 'y_class'

num_columns = ['darksky_temperature', 'darksky_humidity', 'darksky_precipitation_probability', 'sched_hdwy', 'traffic_speed']
cat_columns = ['month', 'hour', 'day', 'stop_sequence', 'stop_id_original', 'year', 'time_window', target]
ohe_columns = ['dayofweek', 'route_id_dir', 'is_holiday', 'is_school_break', 'zero_load_at_trip_end']
    
    
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }
    
def generate_random_trips(n=200, seed=500):
    f = os.path.join('../data', 'processed', 'apc_weather_gtfs.parquet')
    apcdata = spark.read.load(f)
    todelete = apcdata.filter('(load < 0) OR (load IS NULL)').select('transit_date','trip_id','overload_id').distinct()
    todelete=todelete.withColumn('marker',F.lit(1))

    #joining and whereever the records are not found in sync error table the marker will be null
    apcdataafternegdelete=apcdata.join(todelete,on=['trip_id','transit_date','overload_id'],how='left').filter('marker is null').drop('marker')
    apcdataafternegdelete = apcdataafternegdelete.sort(['trip_id', 'overload_id'])

    get_columns = ['trip_id', 'transit_date', 'arrival_time', 
                'route_direction_name', 'route_id', 'block_abbr']
    get_str = ", ".join([c for c in get_columns])

    apcdataafternegdelete.createOrReplaceTempView("apc")

    # # filter subset
    query = f"""
    SELECT {get_str}
    FROM apc
    """

    apcdataafternegdelete = spark.sql(query)
    df = apcdataafternegdelete.toPandas()
    df = df[df.arrival_time.notna()]

    df['route_id_dir'] = df["route_id"].astype("str") + "_" + df["route_direction_name"]
    df['day'] = df["arrival_time"].dt.day
    df = df.sort_values(by=['block_abbr', 'arrival_time']).reset_index(drop=True)
    
    _test_df = df[(df['transit_date'] > test_dates[0]) & (df['transit_date'] < test_dates[1])]
    vc = _test_df.route_id_dir.value_counts()
    common_route_id_dirs = vc[vc >= 2000].index
    # Get unique trips that have more than 10 stops.
    tdf = _test_df[_test_df['route_id_dir'].isin(common_route_id_dirs)].groupby(['transit_date', 'route_id_dir', 'trip_id']).count().reset_index()
    tdf = tdf[tdf.day > 10]

    vc = tdf.route_id_dir.value_counts()
    common_route_id_dirs = vc[vc >= 2000].index
    tdf = tdf[tdf['route_id_dir'].isin(common_route_id_dirs)]
    tdf.route_id_dir.value_counts().plot(kind='bar')

    tdf = tdf.groupby('route_id_dir').sample(n=n, random_state=seed)[['transit_date', 'route_id_dir', 'trip_id']]
    random_trip_df = tdf.drop_duplicates()
    
    fp = os.path.join('../models/same_day/evaluation/random_trip_df_5000.pkl')
    random_trip_df.to_pickle(fp)
    # Sample from route_id_dir grouping equally. Get 500 each

    # How to use:
    # Loop through the random_trip_df and get corresponding data for
    # overall_df.merge(tdf[0:1], how='right', on=['transit_date', 'route_id_dir', 'trip_id'])


def prepare_test_linklevel(test_df, ohe_encoder, num_scaler, label_encoders,
                        cat_columns=None, num_columns=None, ohe_columns=None, feature_label='load'):
    test_df[ohe_encoder.get_feature_names_out()] = ohe_encoder.transform(test_df[ohe_columns]).toarray()
    for col in [col for col in cat_columns if col != feature_label]:
        encoder = label_encoders[col]
        test_df[col] = encoder.transform(test_df[col])
    test_df[num_columns] = num_scaler.transform(test_df[num_columns])
    return test_df


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

def prepare_data(config):
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
    fp = os.path.join(config.holidays_path)
    holidays_df = pd.read_csv(fp)
    holidays_df['Date'] = pd.to_datetime(holidays_df['Date'])
    holidays_df['is_holiday'] = True
    df = df.merge(holidays_df[['Date', 'is_holiday']], left_on='transit_date', right_on='Date', how='left')
    df['is_holiday'] = df['is_holiday'].fillna(False)
    df = df.drop(columns=['Date'])
        
    # School breaks
    fp = os.path.join(config.school_break_path)
    school_break_df = pd.read_pickle(fp)
    school_break_df['is_school_break'] = True
    df = df.merge(school_break_df[['Date', 'is_school_break']], left_on='transit_date', right_on='Date', how='left')
    df['is_school_break'] = df['is_school_break'].fillna(False)
    df = df.drop(columns=['Date'])

    # Traffic
    # Causes 3M data points to be lost
    fp = os.path.join(config.traffic_path)
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
        # Same result as creating a fixed_arrival_time (but faster)d
        ba_df = ba_df[ba_df.stop_sequence != end_stop].reset_index(drop=True)
        sorted_df.append(ba_df)
            
    overall_df = pd.concat(sorted_df)
    drop_cols = ['route_direction_name', 'route_id']
    drop_cols = [col for col in drop_cols if col in overall_df.columns]
    overall_df = overall_df.drop(drop_cols, axis=1)

    TIMEWINDOW = config.time_window
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
                                                        "traffic_speed":"mean",
                                                        "sched_hdwy": "max",
                                                        "load": "sum" })
    overall_df = overall_df.reset_index(level=[0,1,2,3])
    overall_df = overall_df.sort_values(by=['block_abbr', 'arrival_time']).reset_index(drop=True)

    drop_cols = ['arrival_time', 'block_abbr']
    drop_cols = [col for col in drop_cols if col in overall_df.columns]
    overall_df = overall_df.drop(drop_cols, axis=1)
    # checking bins of loads for possible classification problem
    loads = overall_df[overall_df.load <= 100]['load']
    percentiles = []
    class_bins = [0, 33, 66, 100]
    for cbin in class_bins:
        percentile = np.percentile(loads.values, cbin)
        percentiles.append(percentile)

    percentiles = [(percentiles[0], percentiles[1]), (percentiles[1] + 1, percentiles[2]), (percentiles[2] + 1, 55.0), (56.0, 75.0), (76.0, 100.0)]
    logging.info(f"Percentiles: {percentiles}")
    overall_df['y_class'] = overall_df['load'].apply(lambda x: data_utils.get_class(x, percentiles))
    overall_df = overall_df[overall_df['y_class'].notna()]
    
    columns = joblib.load('../models/same_day/LL_X_columns.joblib')
    label_encoders = joblib.load('../models/same_day/LL_Label_encoders.joblib')
    ohe_encoder = joblib.load('../models/same_day/LL_OHE_encoder.joblib')
    num_scaler = joblib.load('../models/same_day/LL_Num_scaler.joblib')

    test_df = deepcopy(overall_df)
    test_df = prepare_test_linklevel(test_df, 
                                    ohe_encoder, num_scaler, label_encoders,
                                    cat_columns=cat_columns,
                                    num_columns=num_columns,
                                    ohe_columns=ohe_columns,
                                    feature_label='y_class')
    
    return test_df

def revere_transform(df, label_encoders, ohe_encoder):
    
    for col in cat_columns:
        if col == 'y_class':
            continue
        df[col] = label_encoders[col].inverse_transform(df[col])
        
    df[ohe_columns] = ohe_encoder.inverse_transform(df.filter(regex='dayofweek_|route_id_dir_|is_holiday_|is_school_break_|zero_load_at_trip_end_'))
    df = df.drop(columns=df.filter(regex='dayofweek_|route_id_dir_|is_holiday_|is_school_|zero_load_').columns, axis=1)
    return df

def evaluate(test_df, config):
    
    label_encoders = joblib.load('../models/same_day/LL_Label_encoders.joblib')
    ohe_encoder = joblib.load('../models/same_day/LL_OHE_encoder.joblib')
    
    # Load same 
    fp = os.path.join('../models', 'same_day', 'LL_X_columns.joblib')
    columns = joblib.load(fp)
    
    # Setup model
    PAST = config.past
    # Load models
    num_features = len(columns)
    simple_lstm = linklevel_utils.setup_simple_lstm_generator(num_features, len(test_df.y_class.unique()))
    # Load model
    latest = tf.train.latest_checkpoint(config.model_path)

    simple_lstm.load_weights(latest)

    # Load random trips for evaluation
    fp = os.path.join(config.test_trips_path)
    random_trip_df = pd.read_pickle(fp)

    results = []
    for i in tqdm(range(len(random_trip_df))):
        trip_df = test_df.merge(random_trip_df.iloc[[i]], on=['transit_date', 'trip_id', 'route_id_dir'])
        if len(trip_df) == 0:
            continue
        inverse_trip_df = deepcopy(trip_df)
        inverse_trip_df = revere_transform(inverse_trip_df, label_encoders, ohe_encoder)
        trip_df = trip_df[columns]
        
        future = len(trip_df) - PAST
        past_df = trip_df.iloc[0:PAST]
        
        inverse_trip_df['y_pred'] = -1

        y_true = trip_df.iloc[PAST:].y_class.tolist()
        y_pred = generate_simple_lstm_predictions(past_df, simple_lstm, future)
        inverse_trip_df.loc[trip_df.iloc[PAST:].index, 'y_pred'] = y_pred
        results.append(inverse_trip_df)
    
    results = pd.concat(results)
    fp = os.path.join(config.path, 'results_5000_df.pkl')
    results.to_pickle(fp)
    
    pass

def evaluate_baseline(test_df, config):
    pass


OUTPUT_DIR = os.path.join('../evaluation', 'same_day', 'clean_evaluation')
def main(configs):
    for config in configs:
        config = dotdict(config)
        OUTPUT_NAME = f"timewindow{config.time_window}"
        OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
        Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
        config.path = OUTPUT_PATH
        logging.info(config)
        
        test_df = prepare_data(config)
        evaluate(test_df, config)
    pass

if __name__ == "__main__":
    log_file = f"../evaluation/same_day/clean_evaluation/{date_str}_gridsearch.log"
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S', 
                        encoding='utf-8',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])
    configs = [
        {
            'past': 5,
            'time_window': 15,
            'model_path':'../models/same_day/school_zero_load',
            'test_trips_path': '../models/same_day/evaluation/random_trip_df_5000.pkl',
            'school_break_path':'../data/others/School Breaks (2019-2022).pkl',
            'holidays_path':'../data/others/US Holiday Dates (2004-2021).csv',
            'traffic_path':'../data/traffic/triplevel_speed.pickle'
        }
    ]
    main(configs)