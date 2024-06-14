# Module Deployment

This module covers various methods for deploying machine learning models, ranging from web services with Flask and Docker to batch scoring and real-time streaming. Below are the different deployment methods and techniques discussed:

## 4.1 Three Ways of Deploying a Model

1. **Containerization with Docker**:
   - Package the machine learning model and its dependencies into a Docker container.
   - Use Dockerfiles to define the environment and build images that encapsulate the model.
   - Ensure portability and reproducibility of the model deployment environment.

2. **Web Services with Flask and Docker**:
   - Deploy models as RESTful APIs using Flask and Docker.
   - Expose endpoints for making predictions over HTTP.
   - Handle incoming requests, perform predictions, and return results.
   - Ensure scalability and fault tolerance in a containerized environment.

## 4.2 Web-Services: Deploying Models with Flask and Docker

- Detailed steps and considerations for deploying machine learning models using Flask and Docker.
- Setting up Flask applications to serve predictions.
- Containerizing Flask applications for deployment in diverse environments.

## 4.3 Web-Services: Getting Models from the Model Registry (MLflow)

- Utilizing MLflow for model registry and versioning.
- Retrieving models from the registry and deploying them as web services.
- Managing model lifecycle including version control and rollback.

## 4.4 (Optional) Streaming: Deploying Models with Kinesis and Lambda

- Real-time model deployment using AWS Kinesis and Lambda.
- Streaming data to Lambda functions for inference and processing.
- Handling continuous data streams and performing predictions in real-time.

## 4.5 Batch: Preparing a Scoring Script

- Creating batch scoring scripts for offline predictions.
- Processing large datasets efficiently using batch inference.
- Saving prediction results to storage systems like AWS S3 or Google Cloud Storage.

