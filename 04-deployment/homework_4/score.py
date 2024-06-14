#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
import pandas as pd
import numpy as np
import os

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Predict trip durations and save results")
    parser.add_argument('--year', type=int, required=True, help='Year of the trip data')
    parser.add_argument('--month', type=int, required=True, help='Month of the trip data')
    return parser.parse_args()

# Function to read and preprocess input data
def read_dataframe(filename):
    df = pd.read_parquet(filename)
    categorical = ['PULocationID', 'DOLocationID']
    
    # Calculate trip duration in minutes
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df['duration'].dt.total_seconds() / 60
    
    # Filter trips based on duration criteria
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)].copy()
    
    # Convert categorical columns to string type
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    # Generate unique ride IDs
    args = parse_args()
    df['ride_id'] = f'{args.year:04d}/{args.month:02d}_' + df.index.astype('str')

    return df

# Function to prepare dictionaries from DataFrame columns
def prepare_dictionaries(df):
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    return dicts

# Function to load the trained model
def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return model, dv

# Function to apply the model for prediction
def apply_model(input_file, output_file):
    # Read data and preprocess
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)
    
    # Load model
    model, dv = load_model()
    
    # Transform data and make predictions
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    
    # Calculate mean predicted duration
    mean_prediction_duration = np.mean(y_pred)
    
    # Print mean predicted duration
    print(f"Mean predicted duration: {mean_prediction_duration:.2f} minutes")

    # Prepare DataFrame for results and save to Parquet file
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['prediction_duration'] = y_pred
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

# Main function to orchestrate the process
def run():
    args = parse_args()
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{args.year:04d}-{args.month:02d}.parquet'
    output_file = f'output/yellow_tripdata_{args.year:04d}-{args.month:02d}.parquet'

    # Apply the model and save results
    apply_model(
        input_file=input_file,
        output_file=output_file
    )

# Entry point of the script
if __name__ == '__main__':
    run()
