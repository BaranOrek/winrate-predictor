
# Using Python 3.12.7 slim image as the base image
FROM python:3.11.9-slim

# Define working directory
WORKDIR /usr/src/myapp

# Copy requirements file
COPY requirements.txt .

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy UI files
COPY . .

# Expose application port
EXPOSE 5000

# Prediction API address
ENV PREDICTOR_API=http://predictor-api-server:5000/winrate_predictor

# Application metadata
LABEL winrate.predictor.ui="0.0.1-beta"

# Run application
CMD ["python3", "app.py"]
