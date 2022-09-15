# Gets traffic data per month per year for all active routes and resamples them into 30 minute time windows
import os
os.chdir("..")
os.chdir("..")
print(os.getcwd())
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np

from tqdm import tqdm
from shapely.geometry import Polygon, LineString
import warnings
warnings.filterwarnings('ignore')
from pyspark import SparkContext,SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql import functions as F
from pyspark import SparkConf
import pandas as pd
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

CORES = cpu_count()

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

filepath = os.path.join(os.getcwd(), "data", "inrix")
inrix_parquet = spark.read.load(filepath)
inrix_parquet.createOrReplaceTempView("inrix")

# filter subset
query = f"""
        SELECT xd_id, measurement_tstamp, speed
        FROM inrix
        """
inrix_parquet = spark.sql(query)
inrix_parquet = inrix_parquet.na.drop(subset=['speed'])
inrix_parquet = inrix_parquet.withColumn('year', F.year(F.col('measurement_tstamp')))
inrix_parquet = inrix_parquet.withColumn('month', F.month(F.col('measurement_tstamp')))

####
def get_df_resample(df):
    output_df = []
    for segment, xd_id_df in df.groupby('xd_id'):
        xd_id_df.index = xd_id_df.measurement_tstamp
        # xd_id_df.drop(columns=['year', 'measurement_tstamp'], inplace=True)
        xd_id_df.drop(columns=['year', 'month', 'measurement_tstamp'], inplace=True)
        xd_id_df = xd_id_df.resample('30min').mean()
        output_df.append(xd_id_df.reset_index())
    if len(output_df) == 0:
        return pd.DataFrame()
    output_df = pd.concat(output_df)
    return output_df

traffic_save_dir = os.path.join('data', 'backup_trip_traffic_df')
print("START")
for year in [2020, 2021, 2022]:
    for month in range(1, 13):
        print(f"{month}-{year}")
        if year == 2022 and month == 4:
            break
        _inrix_parquet = inrix_parquet[(inrix_parquet.month == month) & (inrix_parquet.year == year)]
        _inrix_parquet = _inrix_parquet[_inrix_parquet.xd_id.isin(all_used_segments)]
        inrix_df = _inrix_parquet.toPandas()
        output_df = get_df_resample(inrix_df)
        
        fp = os.path.join(traffic_save_dir, f"inrix_speed_30M_resampled_{year}_{month}.gz")
        output_df.to_parquet(fp, engine='auto', compression='gzip')