from pyspark.sql import functions as F
from sklearn.metrics import mean_squared_error
from .data_utils import *
from sklearn.preprocessing import OneHotEncoder
import math
import joblib
import numpy as np

def get_time_window(row, window):
    minute = row.arrival_time.minute
    minuteByWindow = minute//window
    temp = minuteByWindow + (row.arrival_time.hour * (60/window))
    return math.floor(temp)

def generate_new_features(tdf, time_window=30, past_trips=20, target='y_reg'):
    tdf['day'] = tdf.transit_date.dt.day
    tdf['time_window'] = tdf.apply(lambda x: get_time_window(x, time_window), axis=1)

    sort2 = ['block_abbr', 'transit_date', 'arrival_time', 'route_id_direction']
    tdf = tdf.sort_values(sort2)
    tdf = tdf.dropna()
    return tdf

def generate_new_day_ahead_features(tdf, time_window=30, past_trips=20, target='y_reg'):
    tdf['day'] = tdf.transit_date.dt.day
    tdf['time_window'] = tdf.apply(lambda x: get_time_window(x, time_window), axis=1)

    sort2 = ['block_abbr', 'transit_date', 'arrival_time', 'route_id_direction']
    tdf = tdf.sort_values(sort2)

    tdf['avg_actual_headway_lag'] = tdf['actual_headways'].shift(1)
    tdf['load_lag'] = tdf[target].shift(1)

    tdf['load_pct_change'] = tdf['load_lag'].pct_change(periods=1)
    tdf['act_headway_pct_change'] = tdf['avg_actual_headway_lag'].pct_change(periods=1)

    tdf['avg_past_act_headway'] = tdf['actual_headways'].rolling(window=past_trips, min_periods=1, closed='left').quantile(0.95, interpolation='lower')
    tdf['avg_past_trips_loads'] = tdf[target].rolling(window=past_trips, min_periods=1, closed='left').quantile(0.95, interpolation='lower')

    drop_cols = ['avg_actual_headway_lag', 'load_lag', 'trip_id', 'arrival_time', 'actual_headways']
    drop_cols = [col for col in drop_cols if col in tdf.columns]
    tdf = tdf.drop(drop_cols, axis=1)

    tdf = tdf.dropna()

    # Removing null pct_changes
    tdf.replace([np.inf, -np.inf], np.nan, inplace=True)
    tdf.dropna(subset=["load_pct_change"], how="all", inplace=True)
    return tdf

# Define evaluator, requires some time information (merge testX, testY and pred)
def generate_results_over_time_window(results_df_arr):
    mse_df = pd.DataFrame()
    for i, results_df in enumerate(results_df_arr):
        tdf = results_df[['time_window', 'y_pred', 'y_true']].groupby('time_window').agg({'y_pred': list, 'y_true': list})
        tdf['mse'] = tdf.apply(lambda x: mean_squared_error(x.y_true, x.y_pred, squared=False), axis=1)
        mse_df[i] = tdf['mse']
    return mse_df

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def prepare_df_for_training(df, OHE_COLUMNS, ORD_COLUMNS, add_embedding_id=True, target='y_reg', class_bins=CLASS_BINS):
    df = df[df[target] < TARGET_MAX]
    df, percentiles = add_target_column_classification(df, target, TARGET_COLUMN_CLASSIFICATION, class_bins)
    
    ix_map = {}
    if add_embedding_id:
        for col in ORD_COLUMNS:
            ix_map[col] = create_ix_map(df, df, col)
            df[f"{col}_ix"] = df[col].apply(lambda x: ix_map[col][x])
    df = df.drop(columns=ORD_COLUMNS)
    
    # OHE for route_id_direction
    ohe_encoder = OneHotEncoder()
    ohe_encoder = ohe_encoder.fit(df[OHE_COLUMNS])
    df[ohe_encoder.get_feature_names_out()] = ohe_encoder.transform(df[OHE_COLUMNS]).toarray()

    df = df.drop(columns=OHE_COLUMNS)
    
    return df, ix_map, ohe_encoder, percentiles

# This is hardcoded
def adjust_bins(rf_df, TARGET='y_reg', percentiles=None):
    # Train 2 separate models for bins 0, 1, 2 and 2, 3, 4
    # Adjusting y_class to incorporate Dan's request
    # Use Transit's 3 bins as a base. For the highest capacity bin, carve out everything from 55 to 75 as a 4th bin, and 75+ as a 5th bin.

    percentiles[2] = (16.0, 55.0)
    percentiles.append((56.0, 75.0))
    percentiles.append((76.0, 100.0))

    highload_df = rf_df[(rf_df[TARGET] >= percentiles[3][0]) & (rf_df[TARGET] <= percentiles[3][1])]
    # display(highload_df)
    rf_df.loc[highload_df.index, 'y_class'] = 3

    highload_df = rf_df[(rf_df[TARGET] >= percentiles[4][0]) & (rf_df[TARGET] <= percentiles[4][1])]
    # display(highload_df)
    rf_df.loc[highload_df.index, 'y_class'] = 4
    return rf_df, percentiles

# Convert to the same format as the model input
def prepare_day_ahead_for_prediction(input_df, OHE_COLUMNS):
    cat_features = ['route_id_direction', 'is_holiday', 'dayofweek']
    ord_features = ['year', 'month', 'hour', 'day']
    # num_features = ['temperature', 'humidity', 'precipitation_intensity', 'scheduled_headway', 'time_window']
    #        'load_pct_change',
    #    'act_headway_pct_change', 
    #    'avg_past_act_headway',
    #    'avg_past_trips_loads', 
    train_columns = ['temperature', 'humidity', 'precipitation_intensity',
       'scheduled_headway', 'time_window', 
       'route_id_direction_14_FROM DOWNTOWN',
       'route_id_direction_14_TO DOWNTOWN', 'route_id_direction_17_FROM DOWNTOWN', 'route_id_direction_17_TO DOWNTOWN', 'route_id_direction_18_FROM DOWNTOWN', 'route_id_direction_18_TO DOWNTOWN', 'route_id_direction_19_FROM DOWNTOWN',
       'route_id_direction_19_TO DOWNTOWN', 'route_id_direction_21_NORTHBOUND','route_id_direction_21_SOUTHBOUND','route_id_direction_22_FROM DOWNTOWN','route_id_direction_22_TO DOWNTOWN','route_id_direction_23_FROM DOWNTOWN','route_id_direction_23_TO DOWNTOWN','route_id_direction_24_FROM DOWNTOWN',
       'route_id_direction_24_TO DOWNTOWN', 'route_id_direction_25_NORTHBOUND', 'route_id_direction_25_SOUTHBOUND', 'route_id_direction_28_FROM DOWNTOWN', 'route_id_direction_28_TO DOWNTOWN', 'route_id_direction_29_FROM DOWNTOWN', 'route_id_direction_29_TO DOWNTOWN', 
       'route_id_direction_34_FROM DOWNTOWN', 'route_id_direction_34_TO DOWNTOWN', 'route_id_direction_35_FROM DOWNTOWN', 'route_id_direction_35_TO DOWNTOWN', 'route_id_direction_38_FROM DOWNTOWN', 'route_id_direction_38_TO DOWNTOWN', 'route_id_direction_3_FROM DOWNTOWN',
       'route_id_direction_3_TO DOWNTOWN','route_id_direction_41_FROM DOWNTOWN','route_id_direction_41_TO DOWNTOWN','route_id_direction_42_FROM DOWNTOWN','route_id_direction_42_TO DOWNTOWN','route_id_direction_43_FROM DOWNTOWN','route_id_direction_43_TO DOWNTOWN','route_id_direction_4_FROM DOWNTOWN','route_id_direction_4_TO DOWNTOWN',
       'route_id_direction_50_FROM DOWNTOWN','route_id_direction_50_TO DOWNTOWN','route_id_direction_52_FROM DOWNTOWN','route_id_direction_52_TO DOWNTOWN','route_id_direction_55_FROM DOWNTOWN','route_id_direction_55_TO DOWNTOWN','route_id_direction_56_FROM DOWNTOWN','route_id_direction_56_TO DOWNTOWN','route_id_direction_5_FROM DOWNTOWN',
       'route_id_direction_5_TO DOWNTOWN','route_id_direction_64_FROM RIVERFRONT','route_id_direction_64_TO RIVERFRONT','route_id_direction_6_FROM DOWNTOWN',
       'route_id_direction_6_TO DOWNTOWN', 'route_id_direction_72_EDMONDSON',
       'route_id_direction_72_GRASSMERE', 'route_id_direction_75_NORTHBOUND',
       'route_id_direction_75_SOUTHBOUND', 'route_id_direction_76_LOOP',
       'route_id_direction_79_EASTBOUND', 'route_id_direction_79_NORTHBOUND',
       'route_id_direction_7_FROM DOWNTOWN',
       'route_id_direction_7_TO DOWNTOWN',
       'route_id_direction_84_FROM NASHVILLE',
       'route_id_direction_84_TO NASHVILLE',
       'route_id_direction_86_FROM NASHVILLE',
       'route_id_direction_86_TO NASHVILLE',
       'route_id_direction_8_FROM DOWNTOWN',
       'route_id_direction_8_TO DOWNTOWN', 'route_id_direction_93_LOOP',
       'route_id_direction_94_FROM NASHVILLE',
       'route_id_direction_95_FROM NASHVILLE',
       'route_id_direction_96_FROM NASHVILLE',
       'route_id_direction_96_TO NASHVILLE',
       'route_id_direction_9_FROM DOWNTOWN',
       'route_id_direction_9_TO DOWNTOWN', 'is_holiday_False',
       'is_holiday_True', 'dayofweek_1', 'dayofweek_2', 'dayofweek_3',
       'dayofweek_4', 'dayofweek_5', 'dayofweek_6', 'dayofweek_7', 'year_ix',
       'month_ix', 'hour_ix', 'day_ix']
    
    ix_map = joblib.load('data/TL_IX_map.joblib')
    ohe_encoder = joblib.load('data/TL_OHE_encoders.joblib')
    
    # OHE for route_id_direction
    input_df[ohe_encoder.get_feature_names_out()] = ohe_encoder.transform(input_df[OHE_COLUMNS]).toarray()
    input_df = input_df.drop(columns=OHE_COLUMNS)
    
    # label encode of categorical variables
    for col in ord_features:
        input_df[f'{col}_ix'] = input_df[col].apply(lambda x: ix_map[col][x])
    input_df = input_df.drop(columns=ord_features)
    
    input_df = input_df[train_columns]
    input_df = input_df.dropna()
    input_df[train_columns[0:5]] = input_df[train_columns[0:5]].apply(pd.to_numeric)
    return input_df

def generate_results(input_df):
    results = input_df.groupby('route_id_direction').agg({'y_pred': list, 'time_window': list})
    a = pd.DataFrame(columns=list(range(0, 48)))
    for i, (k, v) in enumerate(results.iterrows()):
        a.loc[i, v['time_window']] = v['y_pred']
    a['route'] = results.index
    a.index = a['route']
    a = a.drop('route', axis=1)
    a = a.apply(pd.to_numeric, errors='coerce')
    a.columns = a.columns.astype('int')
    return a