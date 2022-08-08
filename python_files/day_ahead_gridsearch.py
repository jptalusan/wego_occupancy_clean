import os
# os.chdir("/media/seconddrive/mta_stationing_problem")
import sys
sys.path.insert(0,'..')

print(os.getcwd())
from src.config import *
from pandas.core.common import SettingWithCopyWarning
from src import data_utils, triplevel_utils, day_ahead_prediction_utils

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import joblib

import warnings
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
pd.set_option('display.max_columns', None)
import xgboost as xgb
import importlib
importlib.reload(data_utils)

processed_triplevel = os.path.join('../data', 'processed', 'triplevel_df.parquet')
df = pd.read_parquet(processed_triplevel, engine='auto')
df = df.dropna()
# Removing time_window in case a different one will be used
df = df.drop(['time_window', 'load'], axis=1)
df = df.reset_index(drop=True)
df = df.sort_values(['block_abbr', 'transit_date', 'arrival_time', 'route_id_direction'])

FOLDS = 5
RANDOM_SEED = 100
WINDOW = 30
PAST_TRIPS = 5
TARGET = 'y_reg100'

cat_features = ['route_id_direction', 'is_holiday', 'dayofweek']
ord_features = ['year', 'month', 'hour', 'day']
num_features = ['temperature', 'humidity', 'precipitation_intensity', 'avg_sched_headway', 'time_window', 'traffic_speed',
                'load_pct_change', 'act_headway_pct_change', 'avg_past_act_headway', 'avg_past_trips_loads']

tdf = triplevel_utils.generate_new_day_ahead_features(df, time_window=WINDOW, past_trips=PAST_TRIPS, target=TARGET)
tdf = tdf.groupby(['transit_date', 'route_id_direction', 'time_window']).agg({"year":"first", 
                                                                              "month":"first",
                                                                              "day": "first",
                                                                              "hour":"first",
                                                                              "is_holiday": "first",
                                                                              "dayofweek":"first",
                                                                              "temperature":"mean", 
                                                                              "humidity":"mean",
                                                                              "precipitation_intensity": "mean",
                                                                              "traffic_speed":"mean",
                                                                              "scheduled_headway": "max",
                                                                              "load_pct_change": "max",
                                                                              "act_headway_pct_change": "max",
                                                                              "avg_past_act_headway": "max",
                                                                              "avg_past_trips_loads": "max",
                                                                              TARGET: "max" })
tdf = tdf.reset_index(level=[0,1,2])
print("ohe_encoder is for the following column order:", cat_features)
rf_df, ix_map, ohe_encoder, percentiles = triplevel_utils.prepare_df_for_training(tdf, cat_features, ord_features, target=TARGET)
drop_cols = ['route_id', 'route_direction_name', 'block_abbr', 'y_reg100', 'y_reg095', 'transit_date', 'is_holiday', 'route_id_direction', 'actual_headways', 'trip_id', 'arrival_time']
drop_cols = [col for col in drop_cols if col in rf_df.columns]
rf_df = rf_df.drop(drop_cols, axis=1)

y = rf_df.pop('y_class')
X = rf_df

model = xgb.XGBClassifier(num_class=3, 
                          seed=RANDOM_SEED, 
                          objective='multi:softmax', 
                          eval_metric='mlogloss', 
                          use_label_encoder=False)

skf = StratifiedKFold(n_splits=FOLDS, 
                      random_state=RANDOM_SEED, 
                      shuffle=True)


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
    'gamma': [0.05, 0.1, 0.2]
}

grid_search = RandomizedSearchCV(
    n_iter = 5,
    estimator = model,
    param_distributions=parameters,
    scoring = 'accuracy',
    n_jobs = -1,
    cv = skf,
    verbose=True
)

result = grid_search.fit(X, y)

fp = os.path.join('../models', 'day_ahead', 'XGBOOST_RANDSEARCHCV_2.pkl')
joblib.dump(result, fp)