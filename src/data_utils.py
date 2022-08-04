from src.config import *
import numpy as np
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F

def add_block_tripids(df):
    temp = df[['transit_date', 'block_abbr', 'trip_id']].groupby(by=['transit_date', 'block_abbr']).agg(list).reset_index()
    temp = temp.rename(columns={'trip_id': 'block_trip_ids'})
    df = df.merge(temp, how='left', left_on=['transit_date', 'block_abbr'], right_on=['transit_date', 'block_abbr'], validate="many_to_one")
    return df

def get_percentile(lst, percentile):
    try:
        r = np.percentile(lst, percentile, interpolation='lower')
    except:
        r = None
    return r

def get_average(lst):
    try:
        r = np.mean(lst)
    except:
        r = None
    return r

def get_class(val, percentiles):
    for i, (min, max) in enumerate(percentiles):
        if (val >= min) and (val <= max):
            return i
    # for i in range(len(percentiles)-1):
    #     if (val >= percentiles[i]) and (val <= percentiles[i+1]):
    #         return i
    return None

def create_ix_map(df_train, df_test, col):
    vals = df_train[col].values.tolist() + df_test[col].values.tolist()
    vals = list(set(vals))
    ix_map = {}
    for i in range(len(vals)):
        ix_map[vals[i]] = i
    return ix_map

def median_value_per_trip(df, target_column_regression):
    """
    Returns a dataframe with columns 'trip_id', target_column.
    :param df:
    :param target_column:
    :return:
    """
    temp = df[['trip_id', target_column_regression]].groupby(by=['trip_id']).median()
    temp = temp.rename(columns={target_column_regression: f"{target_column_regression}_expected"})
    return temp

def add_target_column_classification(df, target_column_regression, target_column_classification, class_bins):
    vals = df[target_column_regression].values
    percentiles = [get_percentile(vals, x) for x in class_bins]
    percentiles = [(percentiles[0], percentiles[1]), (percentiles[1] + 1, percentiles[2]), (percentiles[2] + 1, percentiles[3])]
    df[target_column_classification] = df[target_column_regression].apply(lambda x: get_class(x, percentiles))
    return df, percentiles

# y_reg is the loads in percentile
def add_target_column_regression(df, target_column_regression, percentile):
    df[target_column_regression] = df['loads'].apply(lambda x: get_percentile(x, percentile=percentile))
    return df

def remove_nulls_from_apc(apcdata):
    null_arrival_departure_times=apcdata.groupBy('transit_date', 'trip_id','vehicle_id','overload_id','block_abbr').agg((F.sum(F.col('arrival_time').isNull().cast("int")).alias('null_arrival_count')),F.count('*').alias('total_count'))
    null_arrival_departure_times=null_arrival_departure_times.filter('null_arrival_count = total_count').select('transit_date','block_abbr').distinct()
    null_arrival_departure_times=null_arrival_departure_times.withColumn('indicator',F.lit(1))
    delEntries3=apcdata.join(null_arrival_departure_times, on=['transit_date', 'block_abbr'], how='left').filter('indicator is not null').drop('indicator')
    delEntries3=delEntries3.withColumn('remark',F.lit('arrival_times_null'))
    apcdata=apcdata.join(null_arrival_departure_times, on=['transit_date', 'block_abbr'], how='left').filter('indicator is null').drop('indicator')
    todelete = apcdata.filter('(load < 0) OR (load IS NULL)').select('transit_date','trip_id','overload_id').distinct()
    todelete=todelete.withColumn('marker',F.lit(1))

    #joining and whereever the records are not found in sync error table the marker will be null
    apcdataafternegdelete=apcdata.join(todelete,on=['trip_id','transit_date','overload_id'],how='left').filter('marker is null').drop('marker')
    apcdata = apcdataafternegdelete.sort(['trip_id', 'overload_id'])

    todelete = apcdata.filter('arrival_time="NULL" or arrival_time is null').select('transit_date','trip_id','stop_id').distinct()
    todelete=todelete.withColumn('marker',F.lit(1))
    apcdata=apcdata.join(todelete,on=['trip_id','transit_date','stop_id'],how='left').filter('marker is null').drop('marker')

    return apcdata

# route_id route_direction_name, dayofweek, hour/departure_time_str, month, year, darksky_temperature, darksky_humidity
def get_apc_per_trip_sparkview(spark, query=None, window=30):
    """
    Will groupby transit_date, trip_id. Returns a dataframe
    :param spark:
    :return:
    """
    if not query:
        query = """
        SELECT transit_date, trip_id, 
            first(arrival_time) as arrival_time,
            count(stop_id) AS num_records, 
            first(year) AS year, 
            first(month) AS month, 
            first(route_id) AS route_id, 
            first(route_direction_name) AS route_direction_name, 
            first(block_abbr) AS block_abbr, 
            first(dayofweek) as dayofweek, 
            first(hour) as hour,
            mean(FLOAT(darksky_temperature)) as temperature,
            mean(FLOAT(darksky_humidity)) as humidity,
            mean(FLOAT(darksky_precipitation_intensity)) as precipitation_intensity,
            mean(FLOAT(sched_hdwy)) AS scheduled_headway,
            mean(FLOAT(actual_hdwy)) AS actual_headways,
            percentile(INT(load), 1.00) AS y_reg100,
            percentile(INT(load), 0.95) AS y_reg095,
            collect_list(load) AS load
        FROM apcdata
        GROUP BY transit_date, trip_id
        ORDER BY arrival_time
        """

    apcdata_per_trip = spark.sql(query)
    apcdata_per_trip.createOrReplaceTempView('apc_per_trip')
    apcdata_per_trip.cache()

    apcdata_per_trip = apcdata_per_trip.withColumn("route_id_direction", F.concat_ws('_', apcdata_per_trip.route_id, apcdata_per_trip.route_direction_name))

    window = 30.0 # minutes
    apcdata_per_trip = apcdata_per_trip.withColumn("minute", F.minute("arrival_time"))
    apcdata_per_trip = apcdata_per_trip.withColumn("minuteByWindow", apcdata_per_trip.minute/window)
    apcdata_per_trip = apcdata_per_trip.withColumn("time_window", apcdata_per_trip.minuteByWindow + (apcdata_per_trip.hour * (60 / window)))
    apcdata_per_trip = apcdata_per_trip.withColumn("time_window", F.round(apcdata_per_trip.time_window, 0).cast(DoubleType()))
    apcdata_per_trip = apcdata_per_trip.filter(apcdata_per_trip.y_reg100 <= TARGET_MAX)

    # Deleting nulls
    todelete = apcdata_per_trip.filter('temperature="NULL" or temperature is null').select('transit_date','trip_id').distinct()
    todelete = todelete.withColumn('marker',F.lit(1))
    apcdata_per_trip = apcdata_per_trip.join(todelete, on=['trip_id','transit_date'],how='left').filter('marker is null').drop('marker')

    todelete = apcdata_per_trip.filter('scheduled_headway="NULL" or scheduled_headway is null').select('transit_date','trip_id').distinct()
    todelete = todelete.withColumn('marker',F.lit(1))
    apcdata_per_trip = apcdata_per_trip.join(todelete, on=['trip_id','transit_date'],how='left').filter('marker is null').drop('marker')
    cols = "minute", "minuteByWindow", "num_records"
    apcdata_per_trip = apcdata_per_trip.drop(*cols)

    return apcdata_per_trip