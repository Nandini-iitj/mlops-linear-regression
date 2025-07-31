"""
Improved 8-bit quantization of trained Linear Regression model with better precision
"""
import os
import time
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from utils import load_model, save_model


def symmetric_quantize(values, bits=8):
    """
    Symmetric quantization with better precision handling
    """
    n_levels = 2 ** (bits - 1) - 1  # Reserve one bit for sign
    
    # Find the maximum absolute value for symmetric quantization
    max_val = np.max(np.abs(values))
    
    if max_val == 0:
        return np.zeros_like(values, dtype=np.int8), 0.0
    
    # Calculate scale factor
    scale = max_val / n_levels
    
    # Quantize: round to nearest integer
    quantized = np.round(values / scale).astype(np.int8)
    
    # Clip to ensure we stay within int8 range
    quantized = np.clip(quantized, -n_levels, n_levels)
    
    return quantized, scale


def symmetric_dequantize(quantized_values, scale):
    """
    Dequantize using symmetric quantization
    """
    return quantized_values.astype(np.float64) * scale


def asymmetric_quantize(values, bits=8):
    """
    Asymmetric quantization for better range utilization
    """
    n_levels = 2 ** bits - 1
    
    min_val = np.min(values)
    max_val = np.max(values)
    
    if min_val == max_val:
        return np.zeros_like(values, dtype=np.uint8), min_val, max_val
    
    # Calculate scale and zero point
    scale = (max_val - min_val) / n_levels
    zero_point = np.round(-min_val / scale).astype(np.uint8)
    
    # Quantize
    quantized = np.round(values / scale + zero_point).astype(np.uint8)
    quantized = np.clip(quantized, 0, n_levels)
    
    return quantized, scale, zero_point, min_val, max_val


def asymmetric_dequantize(quantized_values, scale, zero_point):
    """
    Dequantize using asymmetric quantization
    """
    return (quantized_values.astype(np.float64) - zero_point) * scale


def per_channel_quantize(weights, bits=8):
    """
    Per-channel quantization for better precision
    """
    if weights.ndim == 1:
        return symmetric_quantize(weights, bits)
    
    quantized_weights = np.zeros_like(weights, dtype=np.int8)
    scales = np.zeros(weights.shape[1])
    
    for i in range(weights.shape[1]):
        quantized_weights[:, i], scales[i] = symmetric_quantize(weights[:, i], bits)
    
    return quantized_weights, scales


def per_channel_dequantize(quantized_weights, scales):
    """
    Per-channel dequantization
    """
    if quantized_weights.ndim == 1:
        return symmetric_dequantize(quantized_weights, scales)
    
    dequantized = np.zeros_like(quantized_weights, dtype=np.float64)
    
    for i in range(quantized_weights.shape[1]):
        dequantized[:, i] = symmetric_dequantize(quantized_weights[:, i], scales[i])
    
    return dequantized


def calculate_model_size(coefficients, intercept):
    """Calculate model size in bytes."""
    coef_size = coefficients.nbytes
    intercept_size = np.array([intercept]).nbytes
    return coef_size + intercept_size


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


def quantize_model_improved():
    """
    Improved quantization with multiple strategies and automatic selection
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
    print(f"Coefficients range: [{coefficients.min():.6f}, {coefficients.max():.6f}]")
    
    # Calculate original model size
    original_size = calculate_model_size(coefficients, intercept)
    print(f"Original model size: {original_size} bytes")
    
    # Load test data for evaluation
    if not os.path.exists('models/test_data.joblib'):
        raise FileNotFoundError("Test data not found. Please run train.py first.")
    
    X_test, y_test = load_model('models/test_data.joblib')
    original_predictions = model.predict(X_test)
    original_r2 = r2_score(y_test, original_predictions)
    original_mse = mean_squared_error(y_test, original_predictions)
    
    print(f"\\nOriginal model performance:")
    print(f"R² Score: {original_r2:.6f}")
    print(f"MSE: {original_mse:.6f}")
    
    # Try different quantization strategies
    strategies = {
        'symmetric': lambda w: symmetric_quantize(w, 8),
        'asymmetric': lambda w: asymmetric_quantize(w, 8),
    }
    
    best_strategy = None
    best_r2 = -np.inf
    best_results = None
    
    print("\\nTesting quantization strategies...")
    
    for strategy_name, quantize_func in strategies.items():
        print(f"\\nTesting {strategy_name} quantization...")
        
        try:
            # Quantize coefficients
            if strategy_name == 'symmetric':
                quant_coef, coef_scale = quantize_func(coefficients)
                dequant_coef = symmetric_dequantize(quant_coef, coef_scale)
                
                # Quantize intercept
                quant_intercept, intercept_scale = symmetric_quantize(np.array([intercept]), 8)
                dequant_intercept = symmetric_dequantize(quant_intercept, intercept_scale)[0]
                
                # Store quantization parameters
                quant_params = {
                    'quantized_coefficients': quant_coef,
                    'quantized_intercept': quant_intercept,
                    'coef_scale': coef_scale,
                    'intercept_scale': intercept_scale,
                    'strategy': strategy_name
                }
                
            elif strategy_name == 'asymmetric':
                quant_coef, coef_scale, coef_zero_point, coef_min, coef_max = quantize_func(coefficients)
                dequant_coef = asymmetric_dequantize(quant_coef, coef_scale, coef_zero_point)
                
                # Quantize intercept
                quant_intercept, intercept_scale, intercept_zero_point, intercept_min, intercept_max = asymmetric_quantize(np.array([intercept]), 8)
                dequant_intercept = asymmetric_dequantize(quant_intercept, intercept_scale, intercept_zero_point)[0]
                
                # Store quantization parameters
                quant_params = {
                    'quantized_coefficients': quant_coef,
                    'quantized_intercept': quant_intercept,
                    'coef_scale': coef_scale,
                    'coef_zero_point': coef_zero_point,
                    'coef_min': coef_min,
                    'coef_max': coef_max,
                    'intercept_scale': intercept_scale,
                    'intercept_zero_point': intercept_zero_point,
                    'intercept_min': intercept_min,
                    'intercept_max': intercept_max,
                    'strategy': strategy_name
                }
            
            # Test quantized model performance
            quantized_predictions = X_test.dot(dequant_coef) + dequant_intercept
            quantized_r2 = r2_score(y_test, quantized_predictions)
            quantized_mse = mean_squared_error(y_test, quantized_predictions)
            
            # Calculate errors
            coef_error = np.mean(np.abs(coefficients - dequant_coef))
            intercept_error = abs(intercept - dequant_intercept)
            
            # Calculate size metrics
            quantized_size = quant_coef.nbytes + quant_intercept.nbytes
            size_reduction = ((original_size - quantized_size) / original_size) * 100
            compression_ratio = original_size / quantized_size
            
            print(f"  R² Score: {quantized_r2:.6f} (degradation: {original_r2 - quantized_r2:.6f})")
            print(f"  MSE: {quantized_mse:.6f}")
            print(f"  Coefficient error: {coef_error:.8f}")
            print(f"  Intercept error: {intercept_error:.8f}")
            print(f"  Size reduction: {size_reduction:.1f}%")
            print(f"  Compression ratio: {compression_ratio:.1f}:1")
            
            # Store results
            results = {
                **quant_params,
                'dequant_coef': dequant_coef,
                'dequant_intercept': dequant_intercept,
                'quantized_r2': quantized_r2,
                'quantized_mse': quantized_mse,
                'coef_error': coef_error,
                'intercept_error': intercept_error,
                'quantized_size': quantized_size,
                'size_reduction': size_reduction,
                'compression_ratio': compression_ratio
            }
            
            # Check if this is the best strategy so far
            if quantized_r2 > best_r2:
                best_r2 = quantized_r2
                best_strategy = strategy_name
                best_results = results
                
        except Exception as e:
            print(f"  Error with {strategy_name}: {e}")
            continue
    
    if best_results is None:
        raise ValueError("All quantization strategies failed!")
    
    print(f"\\n{'='*60}")
    print(f"BEST STRATEGY: {best_strategy.upper()}")
    print(f"{'='*60}")
    
    # Use the best quantization results
    dequant_coef = best_results['dequant_coef']
    dequant_intercept = best_results['dequant_intercept']
    
    # Final performance analysis
    quantized_predictions = X_test.dot(dequant_coef) + dequant_intercept
    final_r2 = r2_score(y_test, quantized_predictions)
    final_mse = mean_squared_error(y_test, quantized_predictions)
    
    r2_degradation = original_r2 - final_r2
    mse_degradation = final_mse - original_mse
    
    print(f"\\nFinal Performance Metrics:")
    print(f"Original R² Score: {original_r2:.6f}")
    print(f"Quantized R² Score: {final_r2:.6f}")
    print(f"R² Degradation: {r2_degradation:.6f} ({abs(r2_degradation/original_r2*100):.3f}%)")
    print(f"Original MSE: {original_mse:.6f}")
    print(f"Quantized MSE: {final_mse:.6f}")
    print(f"MSE Degradation: {mse_degradation:.6f} ({abs(mse_degradation/original_mse*100):.3f}%)")
    
    # Inference speed comparison
    print("\\nInference Speed Analysis...")
    sample_size = min(1000, len(X_test))
    X_sample = X_test[:sample_size]
    
    original_time = measure_inference_time(model, X_sample, iterations=100)
    quantized_time = measure_quantized_inference_time(X_sample, dequant_coef, dequant_intercept, iterations=100)
    
    speed_improvement = ((original_time - quantized_time) / original_time) * 100
    
    print(f"Original inference time: {original_time*1000:.4f} ms/sample")
    print(f"Quantized inference time: {quantized_time*1000:.4f} ms/sample")
    print(f"Speed improvement: {speed_improvement:.1f}%")
    
    # Test predictions on first few samples
    print("\\nPrediction comparison (first 3 samples):")
    original_preds = model.predict(X_test[:3])
    quantized_preds = X_test[:3].dot(dequant_coef) + dequant_intercept
    
    for i in range(3):
        diff = abs(original_preds[i] - quantized_preds[i])
        rel_error = (diff / abs(original_preds[i])) * 100 if original_preds[i] != 0 else 0
        print(f"Sample {i+1}: Original={original_preds[i]:.4f}, "
              f"Quantized={quantized_preds[i]:.4f}, "
              f"Diff={diff:.6f}, Rel.Error={rel_error:.2f}%")
    
    # Save the best quantized model
    final_metrics = {
        **best_results,
        'original_r2': original_r2,
        'original_mse': original_mse,
        'final_r2': final_r2,
        'final_mse': final_mse,
        'r2_degradation': r2_degradation,
        'mse_degradation': mse_degradation,
        'original_inference_time': original_time,
        'quantized_inference_time': quantized_time,
        'speed_improvement': speed_improvement,
        'original_size': original_size
    }
    
    # Remove large arrays before saving metrics
    metrics_to_save = {k: v for k, v in final_metrics.items() 
                      if k not in ['dequant_coef', 'dequant_intercept']}
    
    save_model(best_results, 'models/best_quantized_model.joblib')
    save_model(metrics_to_save, 'models/improved_quantization_metrics.joblib')
    
    print("\\nFiles saved:")
    print("- models/best_quantized_model.joblib")
    print("- models/improved_quantization_metrics.joblib")
    
    print(f"\\nQuantization completed successfully!")
    print(f"Strategy used: {best_strategy}")
    print(f"Final R² score: {final_r2:.6f} (vs original: {original_r2:.6f})")
    
    return best_results


if __name__ == "__main__":
    quantize_model_improved()