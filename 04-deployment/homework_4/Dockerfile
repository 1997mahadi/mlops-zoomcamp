# Dockerfile

# Use the prepared base image
FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Set the working directory inside the container
WORKDIR /app

# Install Pandas (add this line)
RUN pip install pandas fastparquet scikit-learn==1.5.0 pyarrow


# Copy your application script into the container
COPY score.py /app/score.py

# Create the output directory
RUN mkdir output

# Set the command to run your Python script
CMD ["python", "score.py", "--year", "2023", "--month", "4"]
