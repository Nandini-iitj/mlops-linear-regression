# MLOps Linear Regression Pipeline

A complete MLOps pipeline for Linear Regression using the California Housing dataset, featuring model training, quantization, containerization, and automated CI/CD workflows.

## Quick Start

### Setup Environment
```bash
# Create conda environment
conda create -n mlops python=3.9 -y
conda activate mlops
pip3 install -r requirements.txt
```

### Run Pipeline
```bash
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
│   ├── train.py          # Model training pipeline
│   ├── quantize.py       # Manual 8-bit quantization
│   ├── predict.py        # Model inference engine
│   └── utils.py          # Utility functions
├── tests/
│   └── test_train.py     # Comprehensive unit tests
├── .github/workflows/    # CI/CD automation
│   └── ci.yml            # GitHub Actions workflow
├── Dockerfile           # Container configuration
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## Pipeline Components

### Training (src/train.py)
- Loads California Housing dataset from scikit-learn
- Trains LinearRegression model with cross-validation
- Evaluates performance using R² score and MSE
- Persists trained model using joblib serialization
- Achieves 61.26 percent variance explanation on training data

### Testing (tests/test_train.py)
- Dataset Validation: Ensures proper data loading and preprocessing
- Model Architecture: Validates LinearRegression instantiation
- Training Verification: Confirms model coefficients are learned
- Performance Gate: Enforces R² score greater than 0.5 quality threshold
- Regression Analysis: Tests prediction capability on holdout data

### Quantization (src/quantize.py)
- Parameter Extraction: Isolates model coefficients and intercept
- Backup Creation: Saves original float32 parameters
- 8-bit Conversion: Implements manual symmetric quantization to uint8
- Validation: Tests inference accuracy with quantized weights
- Intercept quantized along with coefficients

### Prediction (src/predict.py)
- Loads production-ready trained model
- Executes batch inference on test dataset
- Provides detailed prediction vs actual analysis
- Serves as Docker container entry point

## Docker Container

The containerized solution:
- Installs all Python dependencies automatically
- Runs predict.py by default for immediate results
- Passes CI/CD integration tests
- Enables consistent deployment across environments

## CI/CD Pipeline

Three-stage automated workflow:

1. test_suite: Executes comprehensive pytest suite
2. train_and_quantize: Trains model and applies quantization
3. build_and_test_container: Builds Docker image and validates deployment

Each stage must pass before proceeding to the next, ensuring production readiness.

## Performance Analysis

### Model Performance
| Dataset  | R² Score | MSE Loss |
|----------|----------|----------|
| Training | 0.6126   | 0.5179   |
| Testing  | 0.5758   | 0.5559   |

Dataset Split: 16512 training samples, 4128 test samples

### Quantization Performance Metrics

#### Storage Efficiency
| Metric              | Value |
|---------------------|-------|
| Original Model Size | 72 bytes |
| Quantized Model Size | 9 bytes |
| Size Reduction      | 87.5 percent |
| Compression Ratio   | 8.0 to 1 |

#### Quantization Accuracy
| Metric                  | Value       |
|-------------------------|-------------|
| Average Coefficient Error | 0.00126834 |
| Intercept Error           | 0.00000000 |
| Coefficient Storage       | uint8      |
| Intercept Storage         | uint8      |

#### Model Performance Impact
| Metric     | Original | Quantized | Degradation |
|------------|----------|-----------|-------------|
| R² Score   | 0.5758   | 0.5763    | -0.0005     |
| MSE        | 0.5559   | 0.5552    | -0.0007     |

#### Inference Speed Analysis
| Metric                  | Value        |
|-------------------------|--------------|
| Original Inference Time | 0.0503 ms/sample |
| Quantized Inference Time | 0.0041 ms/sample |
| Speed Improvement       | 91.9 percent |

**Performance Note**: The symmetric quantization strategy improved inference time by over 90 percent with negligible impact on accuracy. This makes it suitable for deployment where speed and size are prioritized over small gains in precision.

## Sample Predictions Analysis

```
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
```

### Quantization Impact on Predictions

```
Prediction comparison (first 3 samples):
Sample 1: Original=0.7191, Quantized=0.7210, Diff=0.001868, Rel.Error=0.26 percent
Sample 2: Original=1.7640, Quantized=1.7643, Diff=0.000320, Rel.Error=0.02 percent
Sample 3: Original=2.7097, Quantized=2.7046, Diff=0.005013, Rel.Error=0.18 percent
```

## Model Architecture

```
Linear Regression Coefficients:
Feature | Coefficient
--------|------------
    1   |     0.854383
    2   |     0.122546
    3   |    -0.294410
    4   |     0.339259
    5   |    -0.002308
    6   |    -0.040829
    7   |    -0.896929
    8   |    -0.869842

Model Intercept: 2.071947
Total Features: 8
```

### Quantization Range Analysis

```
Coefficient Range: [-0.896929, 0.854383]
Intercept Range: [2.071947, 2.071947]
```

## Key Features

- Linear Regression: Scikit-learn implementation with California Housing dataset
- Performance Monitoring: R² score tracking and MSE evaluation
- Manual Quantization: Custom 8-bit compression algorithm with detailed metrics
- Quality Assurance: Unit tests with performance thresholds
- Containerization: Docker-ready deployment
- CI/CD Integration: Automated 3-stage pipeline
- Clean Architecture: Organized codebase with separation of concerns
- Comprehensive Analytics: Storage efficiency, speed improvement, and accuracy trade-off analysis

## Recommendations

Given the updated quantization strategy:

1. Retain symmetric quantization due to minimal accuracy degradation
2. Continue to monitor accuracy bounds for production deployments
3. Explore hybrid or adaptive quantization for further optimization
4. Benchmark on other datasets to validate robustness
5. Log additional stats automatically into future runs for audit and tuning