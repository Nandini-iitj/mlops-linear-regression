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

### Training (`src/train.py`)
- Loads California Housing dataset from scikit-learn
- Trains LinearRegression model with cross-validation
- Evaluates performance using R² score and MSE
- Persists trained model using joblib serialization
- **Key Insight**: Achieves 61.26% variance explanation on training data

### Testing (`tests/test_train.py`)
- **Dataset Validation**: Ensures proper data loading and preprocessing
- **Model Architecture**: Validates LinearRegression instantiation
- **Training Verification**: Confirms model coefficients are learned
- **Performance Gate**: Enforces R² score > 0.5 quality threshold
- **Regression Analysis**: Tests prediction capability on holdout data

### Quantization (`src/quantize.py`)
- **Parameter Extraction**: Isolates model coefficients and intercept
- **Backup Creation**: Saves original float32 parameters
- **8-bit Conversion**: Implements manual quantization to uint8
- **Validation**: Tests inference accuracy with dequantized weights
- **Memory Optimization**: Reduces model footprint by 71%

### Prediction (`src/predict.py`)
- Loads production-ready trained model
- Executes batch inference on test dataset
- Provides detailed prediction vs actual analysis
- Serves as Docker container entry point

## Docker Container

The containerized solution:
- Installs all Python dependencies automatically
- Runs `predict.py` by default for immediate results
- Passes CI/CD integration tests
- Enables consistent deployment across environments

## CI/CD Pipeline

**Three-stage automated workflow:**

1. **`test_suite`**: Executes comprehensive pytest suite (quality gate)
2. **`train_and_quantize`**: Trains model and applies quantization
3. **`build_and_test_container`**: Builds Docker image and validates deployment

*Each stage must pass before proceeding to the next, ensuring production readiness.*

## Performance Analysis

### Model Performance
| Dataset | R² Score | MSE Loss | Interpretation |
|---------|----------|----------|----------------|
| Training | 0.6126 | 0.5179 | Good fit, explains 61% variance |
| Testing | 0.5758 | 0.5559 | Minimal overfitting, generalizes well |

### Quantization Impact Analysis
| Metric | Original (float32) | Quantized (uint8) | Improvement |
|--------|-------------------|-------------------|-------------|
| **Model Size** | 2.1 KB | 0.6 KB | **71% reduction** |
| **Memory Usage** | Baseline | 4x less | **75% savings** |
| **R² Accuracy** | 0.5758 | 0.5758 | **<0.01% loss** |
| **Precision** | 32-bit | 8-bit | **4x compression** |

**Key Insight**: Quantization achieves significant resource savings with negligible accuracy loss, making it ideal for edge deployment.

## Sample Predictions Analysis

```
--- Sample Outputs (first 10) ---
Sample 1: Actual=0.477, Predicted=0.719  | Error: +50.7%
Sample 2: Actual=0.458, Predicted=1.764  | Error: +285.2%
Sample 3: Actual=5.000, Predicted=2.710  | Error: -45.8%
Sample 4: Actual=2.186, Predicted=2.839  | Error: +29.9%
Sample 5: Actual=2.780, Predicted=2.605  | Error: -6.3%
```

**Observation**: Model shows typical linear regression behavior with some outlier predictions, particularly on low-value housing samples.

## Model Architecture

```
Linear Regression Coefficients:
Feature 1 (MedInc):      +0.4487  | Strong positive correlation
Feature 2 (HouseAge):    +0.0097  | Minimal impact
Feature 3 (AveRooms):    -0.1233  | Negative correlation
Feature 4 (AveBedrms):   +0.7831  | Highest positive impact
Feature 5 (Population):  -0.0000  | Negligible effect
Feature 6 (AveOccup):    -0.0035  | Slight negative impact
Feature 7 (Latitude):    -0.4198  | Geographic influence
Feature 8 (Longitude):   -0.4337  | Geographic influence

Intercept: -37.0233
```

## Key Features

- **Linear Regression**: Scikit-learn implementation with California Housing dataset
- **Performance Monitoring**: R² score tracking and MSE evaluation
- **Manual Quantization**: Manual 8-bit compression algorithm
- **Quality Assurance**: Unit tests with performance thresholds
- **Containerization**: Docker-ready deployment
- **CI/CD Integration**: Automated 3-stage pipeline
- **Clean Architecture**: All code in organized directories



