# Homework 4: Deploying Ride Duration Model in Batch Mode

This repository contains the solution to Homework 4 of the MLOps Zoomcamp course. The goal is to deploy a ride duration prediction model using batch processing techniques on Yellow Taxi Trip Records dataset.

## Solution Overview

### 1. Data Exploration and Model Development

#### Jupyter Notebook Development

1. **Jupyter Notebook**: 
   - Developed initial model in a Jupyter Notebook (`score.ipynb`).
   - Explored and cleaned the Yellow Taxi Trip Records dataset for March 2023 data.

2. **Output Generation**: 
   - Used `df_result.to_parquet()` to save the predicted ride durations to a parquet file (`yellow_tripdata_2023-03.parquet`) with PyArrow engine and uncompressed format.

### 2. Notebook to Script Conversion

3. **Notebook Conversion**: 
   - Converted the Jupyter Notebook (`score.ipynb`) to a Python script (`score.py`) using `jupyter nbconvert --to script score.ipynb`.

### 3. Virtual Environment Setup

#### Virtual Environment

4. **Setup Virtual Environment**: 
   - Created a virtual environment using pipenv.
   - Installed required libraries (`pandas`, `scikit-learn==1.4.2`, `fastparquet`) using Pipfile and Pipfile.lock to ensure dependency consistency.

### 4. Deployment in Batch Mode

#### Docker Container

5. **Docker Image Creation**: 
   - Created a Dockerfile based on `agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim`.
   - Installed necessary dependencies (`pandas`, `scikit-learn==1.4.2`, `fastparquet`) to run the script inside the Docker container.

#### File Handling and Output

6. **Output Preparation**: 
   - Generated output dataframe with predicted ride durations in the Python script (`score.py`).
   - Saved the results to a parquet file using pyarrow for efficient storage.

#### Docker Execution

7. **Docker Execution**: 
   - Built the Docker image (`my-score-app`) containing the script and dependencies.
   - Executed the Docker container to predict ride durations for specified months (e.g., March, April, May 2023).

### 5. Results and Validation

#### Validation and Analysis

8. **Validation**: 
   - Validated predictions against the dataset.
   - Calculated metrics such as mean and standard deviation of predicted durations.

#### Final Output

9. **Final Output**: 
   - Ensured that the output file (`yellow_tripdata_2023-03.parquet`) was generated correctly and met size expectations.
