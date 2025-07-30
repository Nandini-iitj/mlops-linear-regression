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
- Achieves 61.26% variance explanation on training data

### Testing (tests/test_train.py)
- Dataset Validation: Ensures proper data loading and preprocessing
- Model Architecture: Validates LinearRegression instantiation
- Training Verification: Confirms model coefficients are learned
- Performance Gate: Enforces R² score > 0.5 quality threshold
- Regression Analysis: Tests prediction capability on holdout data

### Quantization (src/quantize.py)
- Parameter Extraction: Isolates model coefficients and intercept
- Backup Creation: Saves original float32 parameters
- 8-bit Conversion: Implements manual quantization to uint8
- Validation: Tests inference accuracy with dequantized weights
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

1. test_suite: Executes comprehensive pytest suite (quality gate)
2. train_and_quantize: Trains model and applies quantization
3. build_and_test_container: Builds Docker image and validates deployment

Each stage must pass before proceeding to the next, ensuring production readiness.

## Performance Analysis

### Model Performance
| Dataset | R² Score | MSE Loss |
|---------|----------|----------|
| Training | 0.6126 | 0.5179 |
| Testing | 0.5758 | 0.5559 |

Dataset Split: 16,512 training samples, 4,128 test samples (80/20 split)

### Quantization Performance Metrics

#### Storage Efficiency
| Metric | Value |
|--------|-------|
| Original Model Size | 72 bytes |
| Quantized Model Size | 9 bytes |
| Size Reduction | 87.5% |
| Compression Ratio | 8.0:1 |

#### Quantization Accuracy
| Metric | Value |
|--------|-------|
| Average Coefficient Error | 0.00033096 |
| Intercept Error | 0.00145382 |
| Coefficient Storage | uint8 |
| Intercept Storage | uint8 |

#### Model Performance Impact
| Metric | Original | Quantized | Degradation |
|--------|----------|-----------|-------------|
| R² Score | 0.5758 | -0.1817 | 0.7575 (131.6%) |
| MSE | 0.5559 | 1.5485 | 0.9926 (178.6%) |

#### Inference Speed Analysis
| Metric | Value |
|--------|-------|
| Original Inference Time | 0.0957 ms/sample |
| Quantized Inference Time | 0.0083 ms/sample |
| Speed Improvement | 91.3% |

**Performance Note**: While quantization achieves significant storage reduction (87.5%) and inference speed improvement (91.3%), it causes substantial accuracy degradation. The R² score drops from 0.5758 to -0.1817, indicating the quantized model performs worse than a simple mean baseline. This suggests the current 8-bit quantization approach may be too aggressive for this linear regression model.

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
Sample 1: Original=0.7191, Dequantized=1.4977, Diff=0.778571
Sample 2: Original=1.7640, Dequantized=2.6391, Diff=0.875065
Sample 3: Original=2.7097, Dequantized=3.4567, Diff=0.747047
```

## Model Architecture

```
Linear Regression Coefficients:
Feature | Coefficient
--------|------------
    1   |     0.448675
    2   |     0.009724
    3   |    -0.123323
    4   |     0.783145
    5   |    -0.000002
    6   |    -0.003526
    7   |    -0.419792
    8   |    -0.433708

Model Intercept: -37.023278
Total Features: 8
```

### Quantization Range Analysis
```
Coefficient Range: [-0.433708, 0.783145]
Intercept Range: [-37.393510, -36.653045]
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

Given the significant accuracy degradation observed with 8-bit quantization:

1. **Consider 16-bit quantization** for better accuracy-efficiency balance
2. **Implement dynamic range optimization** to better preserve important coefficient values
3. **Add quantization-aware training** to compensate for precision loss
4. **Evaluate per-layer quantization sensitivity** for selective compression
5. **Implement accuracy thresholds** in CI/CD pipeline to prevent deployment of degraded models