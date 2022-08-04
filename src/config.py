import pandas as pd

# models
COMBINE_ROUTE_ID_DIRECTION = True
CONT_COLUMNS = ['start_load', 'humidity', 'precipitation_intensity', 'temperature',	'wind_gust', 'wind_speed', 'y_reg_expected', 'start_load_expected']
if COMBINE_ROUTE_ID_DIRECTION:
    CAT_COLUMNS = ['route_id_direction', 'year', 'month', 'hour', 'dayofweek', 'window_of_day']
else:
    CAT_COLUMNS = ['route_id', 'route_direction_name', 'year', 'month', 'window_of_day']
TARGET_COLUMN_REGRESSION = 'y_reg'
TARGET_COLUMN_CLASSIFICATION = 'y_class'
PERCENTILE = 95
# CLASS_BINS = [0, 25, 50, 75, 100]
CLASS_BINS = [0, 33, 66, 100]
TARGET_MAX = 100
FRAC_TRAIN = 0.8
RANDOM_STATE = 100
train_dates = (pd.to_datetime('2021-11-01', format='%Y-%m-%d'), pd.to_datetime('2021-11-25', format='%Y-%m-%d'))
# val_dates = (pd.to_datetime('2021-11-20', format='%Y-%m-%d'), pd.to_datetime('2021-11-25', format='%Y-%m-%d'))
test_dates = (pd.to_datetime('2021-11-25', format='%Y-%m-%d'), pd.to_datetime('2021-11-30', format='%Y-%m-%d'))

# train_dates = (pd.to_datetime('2021-01-01', format='%Y-%m-%d'), pd.to_datetime('2021-10-31', format='%Y-%m-%d'))
# val_dates = (pd.to_datetime('2021-09-20', format='%Y-%m-%d'), pd.to_datetime('2021-11-25', format='%Y-%m-%d'))
# test_dates = (pd.to_datetime('2021-10-31', format='%Y-%m-%d'), pd.to_datetime('2021-12-31', format='%Y-%m-%d'))

# train_dates = ('2021-01-01', '2021-08-31')
# val_dates = ('2021-09-01', '2021-10-31')
# test_dates = ('2021-11-01', '2021-12-31')

WINDOW_OF_DAY = [
    'late_night', 'late_night', 'late_night', 'late_night', 'late_night', 'late_night',
    'am_rush', 'am_rush', 'am_rush', 
    'am', 'am', 'am', 'am', 'am', 
    'pm_rush', 'pm_rush', 'pm_rush', 
    'pm', 'pm', 'pm',
    'night', 'night',
    'late_night', 'late_night'
]