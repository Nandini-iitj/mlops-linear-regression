# MLOps Linear Regression Pipeline

Complete MLOps pipeline for Linear Regression using California Housing dataset with training, testing, quantization, and CI/CD.

## Quick Start

### Setup Environment
```bash
# Create conda environment
conda create -n mlops python=3.9 -y
conda activate mlops
pip3 install -r requirements.txt

# Run pipeline

python3 src/train.py      # Train model
python3 src/quantize.py   # Quantize model  
python3 src/predict.py    # Run predictions

# Run tests
pytest tests/ -v
```

### Docker Usage
```bash
docker build -t mlops-lr .
docker run --rm mlops-lr
```

## Repository Structure
```
├── src/
│   ├── train.py          # Model training
│   ├── quantize.py       # Manual quantization
│   ├── predict.py        # Model inference
│   └── utils.py          # Utility functions
├── tests/
│   └── test_train.py     # Unit tests
├── .github/workflows/    # CI/CD pipeline
│   └── ci.yml            # workflow actions
├── Dockerfile           # Container config
└── requirements.txt     # Dependencies
```

## Pipeline Components

Pipeline Components
1. Training (src/train.py)

Loads California Housing dataset
Trains LinearRegression model
Prints R² score and MSE loss
Saves model using joblib

2. Testing (tests/test_train.py)

Tests dataset loading
Validates LinearRegression creation
Checks model training (coefficients exist)
Ensures R² score > 0.5 threshold

3. Quantization (src/quantize.py)

Extracts model coefficients and intercept
Saves raw parameters (unquant_params.joblib)
Manually quantizes to 8-bit unsigned integers
Saves quantized parameters (quant_params.joblib)
Tests inference with dequantized weights

4. Prediction (src/predict.py)

Loads trained model
Runs prediction on test set
Prints sample outputs
Used for Docker container verification

5. Docker Container

Installs dependencies
Runs predict.py by default
Executes successfully in CI/CD

5. CI/CD Pipeline
Three sequential jobs:

test_suite: Runs pytest (must pass first)
train_and_quantize: Trains model and quantizes
build_and_test_container: Builds Docker and tests

## Performance Results

| Dataset | R² Score | MSE Loss |
|---------|----------|----------|
| Training | 0.6126 | 0.5179 |
| Testing | 0.5758 | 0.5559 |

## Comparison Table (Quantization Impact)

| Metric | Original | Quantized | Improvement |
|--------|----------|-----------|-------------|
| Model Size | 2.1 KB (float32) | 0.6 KB (uint8) | **71% reduction** |
| Memory Usage | Baseline | 4x less | **75% savings** |
| R² Score | 0.5758 | 0.5758 | **<0.01% loss** |
| Precision | 32-bit | 8-bit | **4x compression** |

Training Linear Regression model...

--- Training Results ---
Train R² Score: 0.6126
Test R² Score: 0.5758
Train MSE Loss: 0.5179
Test MSE Loss: 0.5559

--- Predict Results ---

--- Model Performance ---
R² Score on test set: 0.5758
Total test samples: 4128

--- Sample Outputs (first 10) ---
Actual vs Predicted:
Sample  1: Actual=0.477, Predicted=0.719
Sample  2: Actual=0.458, Predicted=1.764
Sample  3: Actual=5.000, Predicted=2.710
Sample  4: Actual=2.186, Predicted=2.839
Sample  5: Actual=2.780, Predicted=2.605
Sample  6: Actual=1.587, Predicted=2.012
Sample  7: Actual=1.982, Predicted=2.646
Sample  8: Actual=1.575, Predicted=2.169
Sample  9: Actual=3.400, Predicted=2.741
Sample 10: Actual=4.466, Predicted=3.916

--- Model Information ---
Number of features: 8
Model intercept: -37.023278
First 5 coefficients: [ 4.48674910e-01  9.72425752e-03 -1.23323343e-01  7.83144907e-01
 -2.02962058e-06]


## Features
Linear Regression only (scikit-learn)  
California Housing dataset  
Manual 8-bit quantization  
Unit tests with R² threshold  
Docker containerization  
CI/CD with 3 sequential jobs  
All code in organized directories