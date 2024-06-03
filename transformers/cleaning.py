import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def clean_dataframe(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Function to clean the DataFrame
    """
    # Convert pickup and dropoff datetime columns to datetime type
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])

    # Calculate duration in minutes
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

    # Filter trips with duration between 1 and 60 minutes
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]

    # Convert categorical columns to string type
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df
