import argparse
import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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

from tensorflow.keras import backend as K
K.clear_session()

import warnings
import pandas as pd
pd.set_option('display.max_columns', None)
from pandas.core.common import SettingWithCopyWarning

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

num_columns = ['darksky_temperature', 'darksky_humidity', 'darksky_precipitation_probability', 'sched_hdwy', 'traffic_speed']
cat_columns = ['month', 'hour', 'day', 'stop_sequence', 'stop_id_original', 'year', 'time_window', target]
ohe_columns = ['dayofweek', 'route_id_dir', 'is_holiday', 'is_school_break', 'zero_load_at_trip_end']

def setup_data():
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
    print(query)

    apcdataafternegdelete = spark.sql(query)
    apcdataafternegdelete = apcdataafternegdelete.na.fill(value=0,subset=["zero_load_at_trip_end"])
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

    drop_cols = ['transit_date', 'load', 'trip_id', 'arrival_time']
    drop_cols = [col for col in drop_cols if col in train_df.columns]
    train_df = train_df.drop(drop_cols, axis=1)
    val_df = val_df.drop(drop_cols, axis=1)

    arrange_cols = [target] + [col for col in train_df.columns if col != target]
    train_df = train_df[arrange_cols]
    val_df = val_df[arrange_cols]
    
    train_df['y_class'] = train_df.y_class.astype('int')
    val_df['y_class']   = val_df.y_class.astype('int')
    test_df['y_class']  = test_df.y_class.astype('int')
    
    return ohe_encoder, label_encoder, num_scaler, train_df, val_df, test_df

def train(hyperparams, train_df, val_df):
    # ----------------- params -----------------#
    log_file = os.path.join(hyperparams.path, 'log.txt')
    target = 'y_class'
    past = hyperparams.past # Past stops observed
    future = 1 # Future stops predicted
    offset = 0

    learning_rate = hyperparams.learning_rate
    batch_size = hyperparams.batch_size
    epochs = hyperparams.epochs

    # feature_label = config.TARGET_COLUMN_CLASSIFICATION
    patience = 3
    lstm_layer = hyperparams.lstm_layer

    label_index = train_df.columns.tolist().index(target)
    
    label_slice = slice(label_index, label_index + 1, None) # which column the label/labels are
    feature_slice = slice(None, None, None) # Which feature columns are included, by default includes all (even label)
    input_sequence_length = past # number of past information to look at
    output_sequence_length = future # number of time steps to predict
    
    @tf.autograph.experimental.do_not_convert
    def timeseries_dataset_from_dataset(df, feature_slice, label_slice, input_sequence_length, output_sequence_length, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(df.values)
        ds = dataset.window(input_sequence_length + output_sequence_length, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda x: x).batch(input_sequence_length + output_sequence_length)
        
        def split_feature_label(x):
            return x[:input_sequence_length:, feature_slice], x[input_sequence_length:,label_slice]
        
        ds = ds.map(split_feature_label)
        
        return ds.batch(batch_size)

    drop_cols = ohe_columns
    drop_cols = [col for col in drop_cols if col in train_df.columns]
    train_df = train_df.drop(drop_cols, axis=1)
    val_df = val_df.drop(drop_cols, axis=1)

    print(f"Training: {train_df.shape}")
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
    model.add(LSTM(lstm_layer, return_sequences=True))
    model.add(LSTM(lstm_layer))
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
    
    with open(log_file, 'a') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
    checkpoint_filepath = os.path.join(hyperparams.path, 'CLA_cp-epoch{epoch:02d}-loss{val_loss:.2f}.ckpt')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    
    csv_logger = tf.keras.callbacks.CSVLogger(log_file, append=True, separator=';')

    # fit model
    callbacks = [keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True), model_checkpoint_callback, csv_logger]

    history = model.fit(dataset_train, validation_data=dataset_val, epochs=epochs, callbacks=callbacks, verbose=1)
    return history

def evaluation(hyperparams, test_df):
    
    target = 'y_class'
    
    test_df['unique_trip'] = test_df['trip_id'] + '_' + test_df['transit_date'].dt.strftime('%Y-%m-%d')

    fp = os.path.join('../models/same_day/evaluation/random_trip_ids_2000.pkl')
    with open(fp, 'rb') as f:
        random_trip_ids = pickle.load(f)

    drop_cols = ['transit_date', 'load', 'arrival_time', 'trip_id'] + ohe_columns
    drop_cols = [col for col in drop_cols if col in test_df.columns]
    test_df = test_df.drop(drop_cols, axis=1)

    arrange_cols = [target] + [col for col in test_df.columns if col != target]
    test_df = test_df[arrange_cols]

    print(f"Evalulation: {test_df.shape}")
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

    # Load models
    num_features = len(test_df.columns) - 1 # subtract `unique_trip`
    num_classes  = len(test_df.y_class.unique())
    
    simple_lstm = tf.keras.Sequential()
    simple_lstm.add(LSTM(hyperparams.lstm_layer, return_sequences=True))
    simple_lstm.add(LSTM(hyperparams.lstm_layer))
    simple_lstm.add(Dropout(0.2))
    simple_lstm.add(Dense(128, activation='relu'))
    simple_lstm.add(Dropout(0.2))
    simple_lstm.add(Dense(64, activation='relu'))
    simple_lstm.add(Dense(num_classes, activation='softmax'))

    # compile model
    simple_lstm.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=hyperparams.learning_rate),
        metrics=["sparse_categorical_accuracy"],
    )

    input_shape = (None, None, num_features)
    simple_lstm.build(input_shape)
    
    # Load model
    
    latest = tf.train.latest_checkpoint(hyperparams.path)

    print(latest)
    simple_lstm.load_weights(latest)

    PAST = hyperparams.past
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
    fp = os.path.join(hyperparams.path, f'EVAL_{hyperparams.past}_{hyperparams.lstm_layer}_{hyperparams.learning_rate}_{hyperparams.batch_size}.pkl')
    res_df.to_pickle(fp)
    return


# def evaluation_parallel(hyperparams, test_df):
    
#     target = 'y_class'
    
#     test_df['unique_trip'] = test_df['trip_id'] + '_' + test_df['transit_date'].dt.strftime('%Y-%m-%d')

#     fp = os.path.join('../models/same_day/evaluation/random_trip_ids_2000.pkl')
#     with open(fp, 'rb') as f:
#         random_trip_ids = pickle.load(f)

#     drop_cols = ['transit_date', 'load', 'arrival_time', 'trip_id'] + ohe_columns
#     drop_cols = [col for col in drop_cols if col in test_df.columns]
#     test_df = test_df.drop(drop_cols, axis=1)

#     arrange_cols = [target] + [col for col in test_df.columns if col != target]
#     test_df = test_df[arrange_cols]

#     print(f"Evalulation: {test_df.shape}")

#     # Load models
#     num_features = len(test_df.columns) - 1 # subtract `unique_trip`
#     num_classes  = len(test_df.y_class.unique())
    
#     simple_lstm = tf.keras.Sequential()
#     simple_lstm.add(LSTM(hyperparams.lstm_layer, return_sequences=True))
#     simple_lstm.add(LSTM(hyperparams.lstm_layer))
#     simple_lstm.add(Dropout(0.2))
#     simple_lstm.add(Dense(128, activation='relu'))
#     simple_lstm.add(Dropout(0.2))
#     simple_lstm.add(Dense(64, activation='relu'))
#     simple_lstm.add(Dense(num_classes, activation='softmax'))

#     # compile model
#     simple_lstm.compile(
#         loss="sparse_categorical_crossentropy",
#         optimizer=keras.optimizers.Adam(learning_rate=hyperparams.learning_rate),
#         metrics=["sparse_categorical_accuracy"],
#     )

#     input_shape = (None, None, num_features)
#     simple_lstm.build(input_shape)
    
#     # Load model
    
#     latest = tf.train.latest_checkpoint(hyperparams.path)

#     print(latest)
#     simple_lstm.load_weights(latest)

#     PAST = hyperparams.past
#     results = []
    
#     queue = Queue()
#     for random_trip_id in random_trip_ids:
#         queue.put(random_trip_id)
        
#     with Manager() as manager:
#         L = manager.list()
        
#         # Spawn two processes, assigning the method to be executed 
#         # and the input arguments (the queue)
#         processes = [Process(target=process_lsm_prediction, args=(simple_lstm, test_df, PAST, L, queue,)) for _ in range(cpu_count() - 1)]

#         for process in processes:
#             process.start()

#         for process in processes:
#             process.join()

#         res_df = pd.concat(L)

#         # fp = os.path.join(hyperparams.path, f'{hyperparams.time_window}_results.pkl')
#         fp = os.path.join(hyperparams.path, f'EVAL_{hyperparams.time_window}_{hyperparams.past}_{hyperparams.lstm_layer}_{hyperparams.learning_rate}_{hyperparams.batch_size}.pkl')
#         res_df.to_pickle(fp)

# def generate_simple_lstm_predictions(input_df, model, future):
#     predictions = []
#     for f in range(future):
#         pred = model.predict(input_df.to_numpy().reshape(1, *input_df.shape))
#         y_pred = np.argmax(pred)
#         predictions.append(y_pred)
#         last_row = input_df.iloc[[-1]]
#         last_row['y_class'] = y_pred
#         last_row['stop_sequence'] = last_row['stop_sequence'] + 1
#         input_df = pd.concat([input_df[1:], last_row])
#     return predictions

# def process_lsm_prediction(model, test_df, past, L, queue):
#     while queue.qsize() > 0 :
#         random_trip = queue.get()
#         trip_df = test_df[test_df['unique_trip'] == random_trip]
#         drop_cols = ['trip_id', 'transit_date', 'unique_trip']
#         drop_cols = [col for col in drop_cols if col in trip_df.columns]
#         trip_df = trip_df.drop(drop_cols, axis=1)
        
#         future = len(trip_df) - past
#         past_df = trip_df.iloc[0:past]
        
#         y_true = trip_df.iloc[past:].y_class.tolist()
#         y_pred = generate_simple_lstm_predictions(past_df, model, future)
        
#         res_df = pd.DataFrame(np.column_stack(([random_trip]*future, y_true, y_pred)), columns=['trip_id', 'y_true', 'y_pred'])
#         L.append(res_df)
    
def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }

def main(configs):
    logging.info("Start gridsearch...")        
    ohe_encoder, label_encoder, num_scaler, train_df, val_df, test_df = setup_data()
    for config in configs:
        config = dotdict(config)
        OUTPUT_NAME = f"{config.time_window}_{config.past}_{config.lstm_layer}_{config.learning_rate}_{config.batch_size}"
        OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
        Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
        config.path = OUTPUT_PATH
        logging.info(config)
        
        logging.info("Start training...")
        train(config, train_df, val_df)
        logging.info("Start evauating...")
        # evaluation_parallel(config, test_df)
        # evaluation(config, test_df)
    logging.info("Done...")
        
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-p', '--past', type=int, default=10)
    # parser.add_argument('-l', '--lstm_layer', type=int, default=256)
    # parser.add_argument('-r', '--learning_rate', type=float, default=0.001)
    # parser.add_argument('-b', '--batch_size', type=float, default=256)
    # args = parser.parse_args()
    # config = dotdict(namespace_to_dict(args))
    # print(config.past)
    # OUTPUT_NAME = f"{config.past}_{config.lstm_layer}_{config.learning_rate}_{config.batch_size}"
    # print(OUTPUT_NAME)
    # OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    # Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    # config.path = OUTPUT_PATH
    # ohe_encoder, label_encoder, num_scaler, train_df, val_df, test_df = setup_data(config)
    # train(config, train_df, val_df)
    # evaluation(config, test_df)
    
if __name__ == "__main__":        
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S', 
                        filename='../models/same_day/gridsearch/gridsearch.log', 
                        encoding='utf-8')
    
    past = [1, 3, 5, 7, 9, 11]
    time_window = [10, 20, 30, 40, 50, 60]
    # layer = [64, 128, 256]
    # batch_size = [256, 512]
    # learning_rate = [0.1, 0.01, 0.001, 0.0001]
    layer = [128]
    batch_size = [256]
    learning_rate = [0.01]
    epochs = [1]
    configs = [dict(zip(('past',
            'time_window', 
            'lstm_layer',
            'batch_size',
            'learning_rate',
            'epochs'), (_past, 
                        _time_window, 
                        _layer,
                        _batch_size,
                        _learning_rate,
                        _epochs))) for _past, _time_window, _layer, _batch_size, _learning_rate, _epochs in product(past, 
                                                                                                                    time_window, 
                                                                                                                    layer,
                                                                                                                    batch_size,
                                                                                                                    learning_rate,
                                                                                                                    epochs)]
    # configs = [
        # {'past': 5,
        # 'epochs': 10,
        # 'lstm_layer': 128,
        # 'learning_rate': 0.001,
        # 'batch_size': 256},
        
        # {'past': 5,
        # 'epochs': 10,
        # 'lstm_layer': 256,
        # 'learning_rate': 0.001,
        # 'batch_size': 256},
        
        # {'past': 5,
        # 'epochs': 10,
        # 'lstm_layer': 512,
        # 'learning_rate': 0.001,
        # 'batch_size': 256},
        
        # {'past': 3,
        # 'epochs': 10,
        # 'lstm_layer': 256,
        # 'learning_rate': 0.001,
        # 'batch_size': 256},
        
        # {'past': 10,
        # 'epochs': 10,
        # 'lstm_layer': 256,
        # 'learning_rate': 0.001,
        # 'batch_size': 256},
        
        # {'past': 5,
        # 'epochs': 10,
        # 'lstm_layer': 256,
        # 'learning_rate': 0.01,
        # 'batch_size': 256},
        
        # {'past': 5,
        # 'epochs': 10,
        # 'lstm_layer': 256,
        # 'learning_rate': 0.1,
        # 'batch_size': 256},
        
        # {'past': 1,
        # 'epochs': 10,
        # 'lstm_layer': 256,
        # 'learning_rate': 0.1,
        # 'batch_size': 256},
        
        # {'past': 1,
        # 'epochs': 10,
        # 'lstm_layer': 256,
        # 'learning_rate': 0.01,
        # 'batch_size': 256},
        
        # {'past': 1,
        # 'epochs': 10,
        # 'lstm_layer': 256,
        # 'learning_rate': 0.001,
        # 'batch_size': 256}
    # ]

    main(configs)