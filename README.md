# Occupancy Prediction and Stationing Problem

## Environment Setup
1. `cd environment`  
2. `conda create env -f environment.yml`  
3. `conda activate py39`  
4. `pip install -r requirements.txt`  

## Datasets
Extract the datasets to the data folder and into their respective folders.  
* apc: `cleaned-wego-daily.apc.parquet`
* weather: `darksky_nashville_20220406.csv` and `weatherbit_weather_2010_2022.parquet`
* gtfs: `alltrips_mta_wego.parquet`
* traffic: inrix data, can download separately

## Code Setup
1. Merge datasets
    * `notebooks/preprocessing.ipynb`
    * If you want to examine raw GTFS files, see `data/gtfs/raw_gtfs` and you can follow this [article](https://medium.com/analytics-vidhya/the-hitchhikers-guide-to-gtfs-with-python-e9790090952a).
2. Generate models
    * Day ahead (trip level) prediction: `notebooks/day_ahead_prediction.ipynb`
    * Any day ahead (trip level) prediction: `notebooks/any_day_prediction.ipynb`
    * Same day (stop level) prediction: `notebooks/same_day_prediction.ipynb`
3. Evaluate models
    * To follow
4. [Streamlit application](https://github.com/rishavsen1/transit_plot_app) to demonstrate output of models and visualize historical data

## More information
* See `slides` directory (To follow)