"""
Manual quantization of trained Linear Regression model.
"""
import os
import numpy as np
from utils import load_model, save_model, quantize_value, dequantize_value


def quantize_model():
    """
    Quantize trained model parameters to 8-bit integers.
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
    
    # Save quantized parameters
    quantized_params = {
        'quantized_coefficients': quantized_coef,
        'quantized_intercept': quantized_intercept,
        'coef_min': coef_min,
        'coef_max': coef_max,
        'intercept_min': intercept_min,
        'intercept_max': intercept_max
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
    
    # Test inference with dequantized weights
    print("\nTesting inference with dequantized weights...")
    if os.path.exists('models/test_data.joblib'):
        X_test, y_test = load_model('models/test_data.joblib')
        
        # Compare first 3 predictions
        original_pred = model.predict(X_test[:3])
        dequant_pred = X_test[:3].dot(dequant_coef) + dequant_intercept
        
        print("Prediction comparison (first 3 samples):")
        for i in range(3):
            diff = abs(original_pred[i] - dequant_pred[i])
            print(f"Sample {i+1}: Original={original_pred[i]:.4f}, "
                  f"Dequantized={dequant_pred[i]:.4f}, Diff={diff:.6f}")
    
    print("\nQuantization completed successfully!")


if __name__ == "__main__":
    quantize_model()