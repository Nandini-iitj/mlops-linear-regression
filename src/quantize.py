"""
Enhanced manual quantization of trained Linear Regression model 
"""
import os
import time
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from utils import load_model, save_model, quantize_value, dequantize_value


def calculate_model_size(coefficients, intercept):
    """Calculate model size in bytes."""
    coef_size = coefficients.nbytes
    intercept_size = np.array([intercept]).nbytes
    return coef_size + intercept_size


def calculate_r2_with_quantized_model(X, y, dequant_coef, dequant_intercept):
    """Calculate R² score using quantized model parameters."""
    predictions = X.dot(dequant_coef) + dequant_intercept
    return r2_score(y, predictions)


def measure_inference_time(model, X, iterations=1000):
    """Measure average inference time for original model."""
    start_time = time.time()
    for _ in range(iterations):
        _ = model.predict(X)
    end_time = time.time()
    return (end_time - start_time) / iterations


def measure_quantized_inference_time(X, dequant_coef, dequant_intercept, iterations=1000):
    """Measure average inference time for quantized model."""
    start_time = time.time()
    for _ in range(iterations):
        _ = X.dot(dequant_coef) + dequant_intercept
    end_time = time.time()
    return (end_time - start_time) / iterations


def quantize_model():
    """
    Quantize trained model parameters to 8-bit integers with comprehensive analysis.
    """
    # Check if model exists
    if not os.path.exists('models/model.joblib'):
        raise FileNotFoundError("Model not found. Please run train.py first.")
    
    print("Loading trained model...")
    model = load_model('models/model.joblib')
    
    # Extract model parameters
    coefficients = model.coef_
    intercept = model.intercept_
    
    print(f"Model coefficients shape: {coefficients.shape}")
    print(f"Intercept value: {intercept:.6f}")
    
    # Calculate original model size
    original_size = calculate_model_size(coefficients, intercept)
    print(f"Original model size: {original_size} bytes")
    
    # Save raw (unquantized) parameters
    raw_params = {
        'coefficients': coefficients,
        'intercept': intercept
    }
    save_model(raw_params, 'models/unquant_params.joblib')
    
    # Prepare for quantization
    print("\nStarting quantization process...")
    
    # Get min/max values for coefficients
    coef_min = coefficients.min()
    coef_max = coefficients.max()
    
    print(f"Coefficients range: [{coef_min:.6f}, {coef_max:.6f}]")
    
    # For intercept (single value), create a small range
    intercept_range = max(abs(intercept) * 0.01, 1e-6)
    intercept_min = intercept - intercept_range
    intercept_max = intercept + intercept_range
    
    print(f"Intercept range: [{intercept_min:.6f}, {intercept_max:.6f}]")
    
    # Perform quantization
    quantized_coef = quantize_value(coefficients, coef_min, coef_max)
    quantized_intercept = quantize_value(np.array([intercept]), intercept_min, intercept_max)
    
    # Calculate quantized model size
    quantized_size = quantized_coef.nbytes + quantized_intercept.nbytes
    size_reduction = ((original_size - quantized_size) / original_size) * 100
    compression_ratio = original_size / quantized_size
    
    print(f"Quantized model size: {quantized_size} bytes")
    print(f"Size reduction: {size_reduction:.1f}%")
    print(f"Compression ratio: {compression_ratio:.1f}:1")
    
    # Save quantized parameters
    quantized_params = {
        'quantized_coefficients': quantized_coef,
        'quantized_intercept': quantized_intercept,
        'coef_min': coef_min,
        'coef_max': coef_max,
        'intercept_min': intercept_min,
        'intercept_max': intercept_max,
        'original_size': original_size,
        'quantized_size': quantized_size,
        'size_reduction': size_reduction,
        'compression_ratio': compression_ratio
    }
    
    save_model(quantized_params, 'models/quant_params.joblib')
    
    # Test dequantization
    print("\nTesting dequantization accuracy...")
    dequant_coef = dequantize_value(quantized_coef, coef_min, coef_max)
    dequant_intercept = dequantize_value(quantized_intercept, intercept_min, intercept_max)[0]
    
    # Calculate quantization errors
    coef_error = np.mean(np.abs(coefficients - dequant_coef))
    intercept_error = abs(intercept - dequant_intercept)
    
    print(f"Average coefficient error: {coef_error:.8f}")
    print(f"Intercept error: {intercept_error:.8f}")
    
    # Performance analysis with test data
    if os.path.exists('models/test_data.joblib'):
        X_test, y_test = load_model('models/test_data.joblib')
        
        print("\nPerformance Analysis...")
        
        # Original model performance
        original_predictions = model.predict(X_test)
        original_r2 = r2_score(y_test, original_predictions)
        original_mse = mean_squared_error(y_test, original_predictions)
        
        # Quantized model performance
        quantized_predictions = X_test.dot(dequant_coef) + dequant_intercept
        quantized_r2 = r2_score(y_test, quantized_predictions)
        quantized_mse = mean_squared_error(y_test, quantized_predictions)
        
        # Performance degradation
        r2_degradation = original_r2 - quantized_r2
        mse_degradation = quantized_mse - original_mse
        
        print(f"Original R² Score: {original_r2:.6f}")
        print(f"Quantized R² Score: {quantized_r2:.6f}")
        print(f"R² Degradation: {r2_degradation:.6f} ({abs(r2_degradation/original_r2*100):.3f}%)")
        print(f"Original MSE: {original_mse:.6f}")
        print(f"Quantized MSE: {quantized_mse:.6f}")
        print(f"MSE Degradation: {mse_degradation:.6f} ({abs(mse_degradation/original_mse*100):.3f}%)")
        
        # Inference speed comparison
        print("\nInference Speed Analysis...")
        sample_size = min(1000, len(X_test))
        X_sample = X_test[:sample_size]
        
        original_time = measure_inference_time(model, X_sample, iterations=100)
        quantized_time = measure_quantized_inference_time(X_sample, dequant_coef, dequant_intercept, iterations=100)
        
        speed_improvement = ((original_time - quantized_time) / original_time) * 100
        
        print(f"Original inference time: {original_time*1000:.4f} ms/sample")
        print(f"Quantized inference time: {quantized_time*1000:.4f} ms/sample")
        print(f"Speed improvement: {speed_improvement:.1f}%")
        
        # Test inference with dequantized weights (first 3 samples)
        print("\nTesting inference with dequantized weights...")
        print("Prediction comparison (first 3 samples):")
        for i in range(3):
            diff = abs(original_predictions[i] - quantized_predictions[i])
            print(f"Sample {i+1}: Original={original_predictions[i]:.4f}, "
                  f"Dequantized={quantized_predictions[i]:.4f}, Diff={diff:.6f}")
        
        # Save comprehensive metrics
        comprehensive_metrics = {
            **quantized_params,
            'original_r2': original_r2,
            'quantized_r2': quantized_r2,
            'r2_degradation': r2_degradation,
            'original_mse': original_mse,
            'quantized_mse': quantized_mse,
            'mse_degradation': mse_degradation,
            'original_inference_time': original_time,
            'quantized_inference_time': quantized_time,
            'speed_improvement': speed_improvement,
            'coef_error': coef_error,
            'intercept_error': intercept_error
        }
        
        save_model(comprehensive_metrics, 'models/quantization_metrics.joblib')
        print("Saved: models/quantization_metrics.joblib")
    
    print("\nQuantization completed successfully!")


if __name__ == "__main__":
    quantize_model()