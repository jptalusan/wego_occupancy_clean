import os
import pandas as pd
import datetime as dt
from src import data_utils
import numpy as np

### Any day data setup for prediction ###
def convert_pandas_dow_to_pyspark(pandas_dow):
    return (pandas_dow + 1) % 7 + 1

def get_past_data(spark, predict_date):
    DAY_OF_WEEK = convert_pandas_dow_to_pyspark(pd.Timestamp(predict_date).day_of_week)
    filepath = os.path.join(os.getcwd(), "data", "processed", "apc_weather_gtfs.parquet")
    apcdata = spark.read.load(filepath)
    apcdata.createOrReplaceTempView("apc")

    # filter subset
    # TODO: Fix hard coding
    start = '2022-03-06'
    end   =' 2022-04-06'
    query = f"""
            SELECT *
            FROM apc
            WHERE (transit_date >= '{start}') AND (transit_date <= '{end}') AND (dayofweek == '{DAY_OF_WEEK}')
            """
    apcdata=spark.sql(query)
    apcdata = data_utils.remove_nulls_from_apc(apcdata)
    apcdata.createOrReplaceTempView('apcdata')
    apcdata_per_trip = data_utils.get_apc_per_trip_sparkview(spark)
    df = apcdata_per_trip.toPandas()
    df = df.sort_values(by='transit_date')
    
    return df

def load_weather_data(path='data/weather/darksky_nashville_20220406.csv'):
    darksky = pd.read_csv(path)
    # GMT-5
    darksky['datetime'] = darksky['time'] - 18000
    darksky['datetime'] = pd.to_datetime(darksky['datetime'], infer_datetime_format=True, unit='s')
    darksky = darksky.set_index(darksky['datetime'])
    darksky['year'] = darksky['datetime'].dt.year
    darksky['month'] = darksky['datetime'].dt.month
    darksky['day'] = darksky['datetime'].dt.day
    darksky['hour'] = darksky['datetime'].dt.hour
    val_cols= ['temperature', 'humidity', 'precipitation_intensity']
    join_cols = ['year', 'month', 'day', 'hour']
    darksky = darksky[val_cols+join_cols]
    renamed_cols = {k: f"darksky_{k}" for k in val_cols}
    darksky = darksky.rename(columns=renamed_cols)
    darksky = darksky.groupby(['year', 'month', 'day', 'hour']).mean().reset_index()
    return darksky

def load_holiday_data(path='data/others/US Holiday Dates (2004-2021).csv'):
    # Holidays
    holidays_df = pd.read_csv(path)
    holidays_df['Date'] = pd.to_datetime(holidays_df['Date'])
    return holidays_df

def load_speed_data(path='data/traffic/triplevel_speed.pickle'):
    speed_df = pd.read_pickle(path)
    speed_df = speed_df[['time_window', 'route_id_direction', 'traffic_speed']]
    speed_df = speed_df.groupby(['route_id_direction', 'time_window']).mean()
    speed_df = speed_df.reset_index(level=[0,1])
    return speed_df

def load_school_break_data(path='data/others/School Breaks (2019-2022).pkl'):
    return pd.read_pickle(path)
    
def setup_any_day_data(DATE_TO_PREDICT, past_df, darksky, holidays_df, school_break_df, speed_df, TARGET='load'):
    DAY_OF_WEEK = convert_pandas_dow_to_pyspark(pd.Timestamp(DATE_TO_PREDICT).day_of_week)
    a = past_df.groupby(['route_id_direction', 'time_window']).agg({'actual_headways':list, 'scheduled_headway': list, TARGET: list})
    
    a['hour'] = a.index.get_level_values('time_window') // 2
    # TODO: Hack since we don't always have up-to-date weather data for testing
    weather_last_datetime = dt.date(2022, 4, 6)
    if DATE_TO_PREDICT > weather_last_datetime:
        DATE_TO_PREDICT = weather_last_datetime
    d = darksky[['hour', 'darksky_temperature', 'darksky_humidity', 'darksky_precipitation_intensity']]
    d = darksky[(darksky['year']==pd.Timestamp(DATE_TO_PREDICT).year) & 
                (darksky['month']==pd.Timestamp(DATE_TO_PREDICT).month) & 
                (darksky['day']==pd.Timestamp(DATE_TO_PREDICT).day)][['hour', 'darksky_temperature', 'darksky_humidity', 'darksky_precipitation_intensity']]

    a = a.reset_index()
    a = a.merge(d, left_on='hour', right_on='hour').sort_values(by=['route_id_direction', 'time_window'])
    
    # TODO: Speed data is estimate from past or forecast from new dataset
    a = a.merge(speed_df, how='left', 
                left_on=['route_id_direction', 'time_window'], 
                right_on=['route_id_direction', 'time_window'])
    a = a[~a['traffic_speed'].isna()]
    
    a['arrival_time'] = a['time_window'].apply(lambda x: pd.Timestamp(f"{pd.Timestamp(DATE_TO_PREDICT) + pd.Timedelta(str(x * 30) + 'min')}"))
    a['transit_date'] = DATE_TO_PREDICT
    a['transit_date'] = pd.to_datetime(a['transit_date'])
    a['year'] = pd.Timestamp(DATE_TO_PREDICT).year
    a['month'] = pd.Timestamp(DATE_TO_PREDICT).month
    a['day'] = pd.Timestamp(DATE_TO_PREDICT).day
    a['dayofweek'] = DAY_OF_WEEK
    a['is_holiday'] = not holidays_df[holidays_df['Date'] == pd.Timestamp(DATE_TO_PREDICT)].empty
    a['is_school_break'] = not school_break_df[school_break_df['Date'] == pd.Timestamp(DATE_TO_PREDICT)].empty
    a['sched_hdwy95'] = a['scheduled_headway'].apply(lambda x: np.max(x))

    a = a.drop(['actual_headways', 'scheduled_headway', TARGET], axis=1)
    a = a.rename({'darksky_temperature':'temperature', 'darksky_humidity':'humidity', 'darksky_precipitation_intensity': 'precipitation_intensity',
                'sched_hdwy95':'scheduled_headway'}, axis=1)
    a = a.bfill()
    return a

def setup_input_data(DATE_TO_PREDICT, past_df, model_type='any_day'):
    darksky = load_weather_data()
    holidays_df = load_holiday_data()
    school_break_df = load_school_break_data()
    speed_df = load_speed_data()
    input_df = setup_any_day_data(DATE_TO_PREDICT, past_df, darksky, holidays_df, school_break_df, speed_df, TARGET='y_reg100')
    return input_df

# Convert to the same format as the model input
def prepare_any_day_for_prediction(input_df, train_columns, ix_map, ohe_encoder):
    cat_features = ['route_id_direction', 'is_holiday', 'dayofweek', 'is_school_break']
    ord_features = ['year', 'month', 'hour', 'day']
    num_features = ['temperature', 'humidity', 'precipitation_intensity', 'avg_sched_headway', 'time_window', 'traffic_speed']
    
    # OHE for route_id_direction
    input_df[ohe_encoder.get_feature_names_out()] = ohe_encoder.transform(input_df[cat_features]).toarray()
    input_df = input_df.drop(columns=cat_features)
    
    # label encode of categorical variables
    for col in ord_features:
        input_df[f'{col}_ix'] = input_df[col].apply(lambda x: ix_map[col][x])
    input_df = input_df.drop(columns=ord_features)
    
    input_df = input_df[train_columns]
    input_df = input_df.dropna()
    input_df[train_columns[1:6]] = input_df[train_columns[1:6]].apply(pd.to_numeric)
    return input_df

def generate_results(input_df, TIMEWINDOW=30):
    results = input_df.groupby('route_id_direction').agg({'y_pred': list, 'time_window': list})
    a = pd.DataFrame(columns=list(range(0, 24 * (60//TIMEWINDOW))))
    for i, (k, v) in enumerate(results.iterrows()):
        a.loc[i, v['time_window']] = v['y_pred']
    a['route'] = results.index
    a.index = a['route']
    a = a.drop('route', axis=1)
    a = a.apply(pd.to_numeric, errors='coerce')
    a.columns = a.columns.astype('int')
    return a