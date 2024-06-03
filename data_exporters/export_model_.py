import mlflow
import mlflow.sklearn
import pickle
import os

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_model(lr, dv, **kwargs):
    """
    Export the linear regression model and DictVectorizer artifact using MLflow.
    """
    # Set the tracking URI to the MLflow server
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    # Start an MLflow run
    with mlflow.start_run():
        # Log the model
        mlflow.sklearn.log_model(lr, "linear_regression_model")
        
        # Save the DictVectorizer artifact
        dv_path = "dict_vectorizer.pkl"
        with open(dv_path, "wb") as f:
            pickle.dump(dv, f)
        
        # Log the artifact
        mlflow.log_artifact(dv_path, artifact_path="dict_vectorizer")

        # Clean up the local artifact file
        os.remove(dv_path)
