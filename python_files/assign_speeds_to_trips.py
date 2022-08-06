# Gets traffic data per month per year for all active routes and resamples them into 30 minute time windows
import os
os.chdir("..")
os.chdir("..")
print(os.getcwd())
import pandas as pd

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F
from pyspark import SparkConf
import pandas as pd
import pickle
from tqdm import tqdm
import math

spark = SparkSession.builder.config('spark.executor.cores', '8').config('spark.executor.memory', '80g')\
        .config("spark.sql.session.timeZone", "UTC").config('spark.driver.memory', '80g').master("local[26]")\
        .appName("wego-daily").config('spark.driver.extraJavaOptions', '-Duser.timezone=UTC').config('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')\
        .config("spark.sql.datetime.java8API.enabled", "true").config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .config("spark.sql.autoBroadcastJoinThreshold", -1)\
        .config("spark.driver.maxResultSize", 0)\
        .config("spark.shuffle.spill", "true")\
        .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")\
        .config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")\
        .config("spark.ui.showConsoleProgress", "false")\
        .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

fp = os.path.join('data', 'XDSegIDs_for_all_trips.pkl')
with open(fp, 'rb') as f:
    all_used_segments = pickle.load(f)

# Get route linestring data
fp = os.path.join('data', 'route_geoms_df.pkl')
trip_id_geom_data = pd.read_pickle(fp)

# Generate/load 1x1 mile grids
fp = os.path.join('data', '1x1grids_davidson.pkl')
grids_df = pd.read_pickle(fp)
grids_df = grids_df.set_geometry('geometry')

# Load inrix segment data
fp = os.path.join('data', 'inrix_grouped.pkl')
with open(fp, "rb") as fh:
  inrix_segment_df = pickle.load(fh)

inrix_segment_df = inrix_segment_df.set_geometry('geometry')
inrix_segment_df = inrix_segment_df[inrix_segment_df['County_inrix'] == 'davidson']
davidson_segs = inrix_segment_df.XDSegID.unique().tolist()

filepath = os.path.join(os.getcwd(), "data", "inrix")
inrix_parquet = spark.read.load(filepath)
inrix_parquet.createOrReplaceTempView("inrix")

fp = os.path.join('data', 'XDSegIDs_for_all_trips.pkl')
with open(fp, 'rb') as f:
    all_used_segments = pickle.load(f)
all_used_segments = [int(aus) for aus in all_used_segments]

# Load inrix segment data
fp = os.path.join('data', 'inrix_grouped.pkl')
with open(fp, "rb") as fh:
  inrix_segment_df = pickle.load(fh)

inrix_segment_df = inrix_segment_df.set_geometry('geometry')
inrix_segment_df = inrix_segment_df[inrix_segment_df['County_inrix'] == 'davidson']
davidson_segs = inrix_segment_df.XDSegID.unique().tolist()

fp = os.path.join('data', 'triplevel_df_processed_MAIN_NOTEBOOK.pickle')
df = pd.read_pickle(fp)
df = df.dropna()
# Change to the historical average
df['traffic_speed'] = -1.0
df['day'] = df['arrival_time'].dt.day

#######
def get_time_window(row, window):
    minute = row.minute
    minuteByWindow = minute//window
    temp = minuteByWindow + (row.hour * (60/window))
    return math.floor(temp)

def find_grids_intersecting(gdf, linestring):
    spatial_index = gdf.sindex
    possible_matches_index = list(spatial_index.intersection(linestring.bounds))
    possible_matches = gdf.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(linestring)]
    return precise_matches

WINDOW = 30
traffic_save_dir = os.path.join('data', 'backup_trip_traffic_df')

new_df_arr = []

for year in [2020, 2021, 2022]:
    for month in range(1, 13):
        if (year == 2022) and (month >= 3):
            continue
        
        fp = os.path.join(traffic_save_dir, f"inrix_speed_30M_resampled_{year}_{month}.gz")
        traffic_estimate_df = pd.read_parquet(fp, engine='auto')
        traffic_estimate_df['time_window'] = traffic_estimate_df['measurement_tstamp'].apply(lambda x: get_time_window(x, WINDOW))
        traffic_estimate_df['day'] = traffic_estimate_df['measurement_tstamp'].dt.day
        
        if traffic_estimate_df.empty:
            continue
        
        _df = df[(df['year']==year) & (df['month']==month)]
        pbar = tqdm(total=len(_df))
        for k, v in _df.iterrows():
            trip_id = v['trip_id']
            day = v['day']
            time_window = v['time_window']
            transit_date = v['transit_date']
            
            route_linestring = trip_id_geom_data[trip_id_geom_data['trip_id'] == trip_id]['geometry'].values[0]
            if route_linestring is None: 
                print("trip id LS not found.")
                continue
            
            route_grids = find_grids_intersecting(grids_df, route_linestring)
            if route_grids.empty: 
                print("route grids for trip not found.")
                continue
            
            route_segments = inrix_segment_df[inrix_segment_df['geometry'].within(route_grids.unary_union)]['XDSegID'].tolist()
            if len(route_segments) == 0: 
                print("route segments for trip not found.")
                continue
            
            _tedf = traffic_estimate_df[(traffic_estimate_df['day'] == day) & (traffic_estimate_df['time_window'] == time_window)]
            _tedf = _tedf[_tedf['xd_id'].isin(route_segments)]
            speed_mean = _tedf.speed.mean()
            _df.at[k, "traffic_speed"] = speed_mean
            pbar.update(1)
        pbar.close()
        
        fp = os.path.join(traffic_save_dir, f"backup_processed_df_{year}_{month}.gz")
        _df.to_parquet(fp, engine='auto', compression='gzip')
        new_df_arr.append(_df)
        
new_df_arr = pd.concat(new_df_arr)
fp = os.path.join('data', 'triplevel_df_processed_with_traffic.pickle')
new_df_arr.to_pickle(fp)