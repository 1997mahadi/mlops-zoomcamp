import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

@data_loader
def ingest_yellow_taxi_data(**kwargs) -> pd.DataFrame:
    """
    Function to load March 2023 Yellow taxi trips data
    """
    # URL for the March 2023 Yellow taxi trips data
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
    
    # Read the data from the provided URL into a DataFrame
    df = pd.read_parquet(url)
    
    return df
