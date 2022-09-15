import logging
import os
import pandas as pd
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
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit
import datetime as dt
import swifter
import numpy as np
import argparse
# Requires the preprocessed dataset `triplevel_df.parquet`

OUTPUT_DIR = os.path.join('../models', 'any_day', 'variable_timewindow')
ohe_features = ['route_id_direction', 'is_holiday', 'dayofweek', 'is_school_break', 'time_window']
ord_features = ['year', 'month', 'hour', 'day']
num_features = ['temperature', 'humidity', 'precipitation_intensity', 'avg_sched_headway', 'traffic_speed']

date_str = dt.datetime.today().strftime('%Y-%m-%d')
    
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
    
def get_statistical_prediction(df, row, percentile, lookback_duration, TARGET='y_reg100'):
    trip_id = row.trip_id
    transit_date = row.transit_date
    route_id_direction = row.route_id_direction
    lookback_date = transit_date - pd.Timedelta(lookback_duration)
    tdf = df[(df['transit_date'] >= lookback_date) & \
             (df['transit_date'] < transit_date)]
    tdf = tdf[(tdf['trip_id'] == trip_id) & \
              (tdf['route_id_direction'] == route_id_direction)]
    if tdf.empty:
        return -1
    return np.percentile(tdf[TARGET].to_numpy(), percentile)

# TODO: Add trip id back
def load_data(config):
    logging.info("Start loading data.")
    processed_triplevel = os.path.join('../data', 'processed', 'triplevel_df.parquet')
    df = pd.read_parquet(processed_triplevel, engine='auto')
    df = df.dropna()
    # Removing time_window in case a different one will be used
    df = df.drop(['time_window', 'load'], axis=1)
    df = df.reset_index(drop=True)
    df = df.sort_values(['block_abbr', 'transit_date', 'arrival_time', 'route_id_direction'])
    # print(df.head(1))
    logging.debug(df.columns.tolist())
    tdf = triplevel_utils.generate_new_features(df, time_window=config.time_window, past_trips=config.past_trips, target=config.target)
    tdf = tdf.groupby(['transit_date', 'route_id_direction', 'time_window']).agg({"trip_id":"first",
                                                                                  "year":"first", 
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
                                                                                  config.target: "max" })
    tdf = tdf.reset_index(level=[0,1,2])
    rf_df, ix_map, ohe_encoder, percentiles = triplevel_utils.prepare_df_for_training(tdf, ohe_features, ord_features, target=config.target)
    rf_df, percentiles = triplevel_utils.adjust_bins(rf_df, TARGET=config.target, percentiles=percentiles)

    original_rf = deepcopy(rf_df)
    original_rf['time_window'] = tdf['time_window']
    drop_cols = ['time_window', 'route_id', 'route_direction_name', 'block_abbr', 'y_reg100', 'y_reg095', 'transit_date', 'is_holiday', 'route_id_direction', 'actual_headways', 'trip_id', 'arrival_time']
    drop_cols = [col for col in drop_cols if col in rf_df.columns]
    rf_df = rf_df.drop(drop_cols, axis=1)

    y = rf_df.pop('y_class')
    X = rf_df

    fp = os.path.join(config.path, 'TL_OHE_encoders.joblib')
    joblib.dump(ohe_encoder, fp)
    fp = os.path.join(config.path, 'TL_IX_map.joblib')
    joblib.dump(ix_map, fp)
    fp = os.path.join(config.path, 'TL_X_columns.joblib')
    joblib.dump(X.columns, fp)
    
    return original_rf, X, y, ix_map, ohe_encoder, percentiles

# Meant for grid search
def train(X, y, config):
    logging.info("Start training.")
    model = xgb.XGBClassifier(num_class=config.num_class, 
                              seed=config.random_seed, 
                              objective='multi:softmax', 
                              eval_metric='mlogloss', 
                              use_label_encoder=False)

    sss = StratifiedShuffleSplit(n_splits=config.folds, test_size=config.test_size, random_state=config.random_seed)

    # parameters = {
    #     'max_depth': [1],
    #     'n_estimators': [1],
    #     'learning_rate': [1],
    #     'gamma': [1]
    # }

    parameters = {
        'max_depth': list(range (2, 24, 6)),
        'n_estimators': list(range(100, 1100, 400)),
        'learning_rate': [0.1, 0.01, 0.05, 0.005],
        'gamma': [0, 0.05, 0.1, 0.2]
    }

    grid_search = RandomizedSearchCV(
        n_iter = 5,
        estimator = model,
        param_distributions=parameters,
        scoring = 'accuracy',
        n_jobs = -1,
        cv = sss,
        verbose=True
    )
    
    result = grid_search.fit(X, y)

    fp = os.path.join(config.path, f'{date_str}_XGBOOST_RANDSEARCHCV.joblib')
    joblib.dump(result, fp)

def evaluate_baseline(config):
    logging.info("Start baseline evaluation...")
    processed_triplevel = os.path.join('../data', 'processed', 'triplevel_df.parquet')
    df = pd.read_parquet(processed_triplevel, engine='auto')
    df = df.dropna()
    # Removing time_window in case a different one will be used
    df = df.drop(['time_window', 'load'], axis=1)
    df = df.reset_index(drop=True)
    df = df.sort_values(['block_abbr', 'transit_date', 'arrival_time', 'route_id_direction'])

    percentiles = [(0.0, 9.0), (10.0, 16.0), (17.0, 55.0), (56.0, 75.0), (76.0, 100.0)]

    df['minute'] = df['arrival_time'].dt.minute
    df['minuteByWindow'] = df['minute'] // config.time_window
    df['temp'] = df['minuteByWindow'] + (df['hour'] * 60 / config.time_window)
    df['time_window'] = np.floor(df['temp']).astype('int')
    df = df.drop(columns=['minute', 'minuteByWindow', 'temp'], axis=1)

    df = df.groupby(['transit_date', 'route_id_direction', 'time_window']).agg({"trip_id":"first",
                                                                                  "year":"first", 
                                                                                  "month":"first",
                                                                                  "arrival_time":"first",
                                                                                  config.target: "max" })
    df = df.reset_index(level=[0,1,2])
    
    df['y_class'] = df[config.target].swifter.apply(lambda x: data_utils.get_class(x, percentiles))
    df['y_class'] = df['y_class'].astype('int')
    
    X, y = df[['transit_date', 'trip_id', 'arrival_time', 'route_id_direction', 'time_window', config.target]], df['y_class']

    sss = StratifiedShuffleSplit(n_splits=config.folds, test_size=config.test_size, random_state=config.random_seed)
    sss.get_n_splits(X, y)
    # skf = StratifiedKFold(n_splits=config.folds, random_state=config.random_seed, shuffle=True)
    # skf.get_n_splits(X, y)

    lookback_distances = ['4W', '2W', '1W']
    percentile = 1.0
    results_df_arr = []
    for _, test_index in sss.split(X, y):
        for lookback_distance in lookback_distances:
            baseline_X = X.iloc[test_index]
            baseline_Y = y.iloc[test_index]
            
            baseline_X['y_pred'] = baseline_X.swifter.apply(lambda x: get_statistical_prediction(df, x, percentile, lookback_distance, TARGET=config.target), axis=1)
            baseline_X['y_true'] = baseline_Y.to_numpy()
            results_df_arr.append(baseline_X)
        break

    results_df_arr[0]['y_pred_class'] = results_df_arr[0]['y_pred'].apply(lambda x: data_utils.get_class(x, percentiles))
    results_df_arr[1]['y_pred_class'] = results_df_arr[1]['y_pred'].apply(lambda x: data_utils.get_class(x, percentiles))
    results_df_arr[2]['y_pred_class'] = results_df_arr[2]['y_pred'].apply(lambda x: data_utils.get_class(x, percentiles))
    df1 = results_df_arr[0].dropna(subset=['y_pred_class'])
    df2 = results_df_arr[1].dropna(subset=['y_pred_class'])
    df3 = results_df_arr[2].dropna(subset=['y_pred_class'])
    df1 = df1.rename(columns={'y_pred_class': 'y_pred'})
    df2 = df2.rename(columns={'y_pred_class': 'y_pred'})
    df3 = df3.rename(columns={'y_pred_class': 'y_pred'})
    df1['past'] = 1
    df2['past'] = 2
    df3['past'] = 4

    fp = os.path.join(config.path, f'{date_str}_baseline_raw_results.pkl')
    pd.concat([df1, df2, df3]).to_pickle(fp)

def reconstruct_original_data(df, ix_map, ohe_encoder):
    df[ohe_features] = ohe_encoder.inverse_transform(df.filter(regex='route_id_direction_|is_holiday_|dayofweek_|is_school_break_|time_window_'))
    
    for col in ord_features:
        inv_map = {v: k for k, v in ix_map[col].items()}
        df[col] = df[f"{col}_ix"].apply(lambda x: inv_map[x])
        
    df = df.drop(columns=df.filter(regex='route_id_direction_|is_holiday_|dayofweek_|is_school_break_|time_window_|_ix').columns, axis=1)
    logging.debug(f"Reconstructed: {df.columns}")
    return df

# Stand alone evaluation, can work better if xgboost params are provided
def evaluate(config, original_rf, X, y, ix_map, ohe_encoder):    
    logging.info("Start evaluation.")
    objective = 'multi:softmax'
    sss = StratifiedShuffleSplit(n_splits=config.folds, test_size=config.test_size, random_state=config.random_seed)
    sss.get_n_splits(X, y)
    
    columns = X.columns
    
    reconstructed_df_arr = []
    kfold = 0
    for train_index, test_index in sss.split(X, y):
        _X_train, _X_test = X.iloc[train_index], X.iloc[test_index]
        _y_train, _y_test = y.iloc[train_index], y.iloc[test_index]
        
        # TODO: or use pre-gridsearched params
        if config.xgboost_params:
            model = xgb.XGBClassifier(n_estimators=config.xgboost_params.n_estimators, 
                                      max_depth=config.xgboost_params.max_depth,
                                      learning_rate=config.xgboost_params.learning_rate, 
                                      use_label_encoder=False,
                                      gamma=config.xgboost_params.gamma, num_class=config.num_classes,
                                      objective=objective, eval_metric='mlogloss')
        else:
            model = xgb.XGBClassifier(use_label_encoder=False, 
                                      objective=objective, 
                                      eval_metric='mlogloss', 
                                      num_class=config.num_classes)
            
        model.fit(_X_train, _y_train, verbose=1)

        preds = model.predict(_X_test)
        _X_test['y_pred'] = preds
        _X_test['y_true'] = _y_test
        _original_rf = original_rf.iloc[test_index]
        _original_rf['y_pred'] = preds
        _original_rf['y_true'] = _y_test
        
        reconstructed_X_test = reconstruct_original_data(_original_rf, ix_map, ohe_encoder)
        reconstructed_X_test['kfold'] = kfold
        kfold = kfold + 1
        reconstructed_df_arr.append(reconstructed_X_test)
    
    reconstructed_df_arr = pd.concat(reconstructed_df_arr)
    output_path = os.path.join(config.path, f'{date_str}_evaluation_results_df.pkl')
    reconstructed_df_arr.to_pickle(output_path)
    return reconstructed_df_arr

def main(configs):
    logging.info("Start gridsearch...")
    for config in configs:
        config = dotdict(config)
        OUTPUT_NAME = f"any_day_tw_{config.time_window}"
        OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
        Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
        config.path = OUTPUT_PATH
        logging.info(config)
        
        # original_rf, X, y, ix_map, ohe_encoder, percentiles = load_data(config)
        # train(X, y, config)
        # logging.info(f"Percentiles: {percentiles}.")
        # reconstructed_df_arr = evaluate(config, original_rf, X, y, ix_map, ohe_encoder)
        # logging.debug(reconstructed_df_arr)
        
        # evaluate_baseline(config)
    
if __name__ == "__main__":
    log_file = f"../models/any_day/variable_timewindow/{date_str}_gridsearch.log"
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S', 
                        encoding='utf-8',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])
    FOLDS = 3
    NUM_CLASSES = 5
    
    configs = [
        {
         'time_window': 1,
         'xgboost_params': None,
         'folds': FOLDS,
         'random_seed': 100,
         'past_trips': 5,
         'target': 'y_reg100',
         'test_size': 0.3,
         'num_classes': NUM_CLASSES
        },
        {
         'time_window': 5,
         'xgboost_params': None,
         'folds': FOLDS,
         'random_seed': 100,
         'past_trips': 5,
         'target': 'y_reg100',
         'test_size': 0.3,
         'num_classes': NUM_CLASSES
        },
        {
         'time_window': 10,
         'xgboost_params': None,
         'folds': FOLDS,
         'random_seed': 100,
         'past_trips': 5,
         'target': 'y_reg100',
         'test_size': 0.3,
         'num_classes': NUM_CLASSES
        },
        {
         'time_window': 20,
         'xgboost_params': None,
         'folds': FOLDS,
         'random_seed': 100,
         'past_trips': 5,
         'target': 'y_reg100',
         'test_size': 0.3,
         'num_classes': NUM_CLASSES
        },
        {
         'time_window': 30,
         'xgboost_params': None,
         'folds': FOLDS,
         'random_seed': 100,
         'past_trips': 5,
         'target': 'y_reg100',
         'test_size': 0.3,
         'num_classes': NUM_CLASSES
        },
        {
         'time_window': 40,
         'xgboost_params': None,
         'folds': FOLDS,
         'random_seed': 100,
         'past_trips': 5,
         'target': 'y_reg100',
         'test_size': 0.3,
         'num_classes': NUM_CLASSES
        },
        {
         'time_window': 50,
         'xgboost_params': None,
         'folds': FOLDS,
         'random_seed': 100,
         'past_trips': 5,
         'target': 'y_reg100',
         'test_size': 0.3,
         'num_classes': NUM_CLASSES
        },
        {
         'time_window': 60,
         'xgboost_params': None,
         'folds': FOLDS,
         'random_seed': 100,
         'past_trips': 5,
         'target': 'y_reg100',
         'test_size': 0.3,
         'num_classes': NUM_CLASSES
        },
        {
         'time_window': 120,
         'xgboost_params': None,
         'folds': FOLDS,
         'random_seed': 100,
         'past_trips': 5,
         'target': 'y_reg100',
         'test_size': 0.3,
         'num_classes': NUM_CLASSES
        },
    ]
    main(configs)