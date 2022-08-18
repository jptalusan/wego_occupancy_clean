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
2. Generate additional data
    * `notebook/data_generation.ipynb`: If some datasets are still missing, please contact me.
    * `notebook/traffic_data.ipynb`: Requires inrix data and might take a long time, i just use speed estimates i previously generated: `data/traffic/triplevel_speed.pickle`
3. Generate models
    * Day ahead (trip level) prediction: `notebooks/day_ahead_prediction.ipynb`
    * Any day ahead (trip level) prediction: `notebooks/any_day_prediction.ipynb`
    * Same day (stop level) prediction: `notebooks/same_day_prediction.ipynb`
    * Boarding/Alighting (stop level) prediction: `notebooks/ons_offs_models.ipynb` (not yet evaluated)
4. [Dash application](https://github.com/jptalusan/mta_carta_dashboard) to demonstrate output of models and visualize historical data

## More information
* See `slides` directory (To follow)