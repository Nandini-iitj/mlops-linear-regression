# Use official Python 3.9 slim image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src/ ./src/
COPY models/ ./models/

# Create models directory if it doesn't exist
RUN mkdir -p models

# Set Python path so imports work correctly
ENV PYTHONPATH=/app/src

# Default command to run predict.py
CMD ["python", "src/predict.py"]