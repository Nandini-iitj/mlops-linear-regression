"""
Improved utility functions for model quantization and data loading
"""
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(test_size=0.2, random_state=42, scale_features=True):
    """
    Load and preprocess California housing dataset.
    
    Args:
        test_size (float): Proportion of dataset to include in test split
        random_state (int): Random state for reproducible splits
        scale_features (bool): Whether to standardize features
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Load California housing dataset
    print("Loading California housing dataset...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features if requested
    if scale_features:
        print("Scaling features...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    print(f"Dataset loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train, y_test


def load_model(filepath):
    """Load a model from file."""
    return joblib.load(filepath)


def save_model(model, filepath):
    """Save a model to file."""
    joblib.dump(model, filepath)
    print(f"Saved: {filepath}")


def quantize_value(values, min_val, max_val, bits=8):
    """
    Original quantization function (for compatibility)
    """
    # Convert to numpy array if not already
    values = np.asarray(values)
    
    # Calculate the range
    value_range = max_val - min_val
    
    # Avoid division by zero
    if value_range == 0:
        return np.zeros_like(values, dtype=np.uint8)
    
    # Calculate the scale factor
    scale_factor = (2**bits - 1) / value_range
    
    # Quantize the values
    quantized = np.round((values - min_val) * scale_factor).astype(np.uint8)
    
    return quantized


def dequantize_value(quantized_values, min_val, max_val, bits=8):
    """
    Original dequantization function (for compatibility)
    """
    # Convert to numpy array if not already
    quantized_values = np.asarray(quantized_values)
    
    # Calculate the range
    value_range = max_val - min_val
    
    # Calculate the scale factor
    scale_factor = (2**bits - 1) / value_range
    
    # Dequantize the values
    dequantized = (quantized_values.astype(np.float64) / scale_factor) + min_val
    
    return dequantized


# New improved quantization functions
def improved_symmetric_quantize(values, bits=8):
    """
    Improved symmetric quantization with better handling of edge cases
    """
    values = np.asarray(values, dtype=np.float64)
    
    # Use (2^(bits-1) - 1) levels for signed quantization
    max_int = 2**(bits-1) - 1
    min_int = -max_int
    
    # Find the maximum absolute value
    abs_max = np.max(np.abs(values))
    
    if abs_max == 0:
        return np.zeros_like(values, dtype=np.int8), 1.0
    
    # Calculate scale factor
    scale = abs_max / max_int
    
    # Quantize and clip
    quantized = np.round(values / scale)
    quantized = np.clip(quantized, min_int, max_int).astype(np.int8)
    
    return quantized, scale


def improved_symmetric_dequantize(quantized_values, scale):
    """
    Improved symmetric dequantization
    """
    return quantized_values.astype(np.float64) * scale


def improved_asymmetric_quantize(values, bits=8):
    """
    Improved asymmetric quantization using the full range
    """
    values = np.asarray(values, dtype=np.float64)
    
    min_val = np.min(values)
    max_val = np.max(values)
    
    if min_val == max_val:
        return np.zeros_like(values, dtype=np.uint8), 1.0, 0, min_val, max_val
    
    # Use full range for unsigned quantization
    qmin, qmax = 0, 2**bits - 1
    
    # Calculate scale and zero point
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point_real = qmin - min_val / scale
    zero_point = int(np.round(np.clip(zero_point_real, qmin, qmax)))
    
    # Quantize
    quantized = np.round(values / scale + zero_point)
    quantized = np.clip(quantized, qmin, qmax).astype(np.uint8)
    
    return quantized, scale, zero_point, min_val, max_val


def improved_asymmetric_dequantize(quantized_values, scale, zero_point):
    """
    Improved asymmetric dequantization
    """
    return (quantized_values.astype(np.float64) - zero_point) * scale


def validate_quantization(original, quantized, dequantized, tolerance=1e-3):
    """
    Validate quantization quality
    """
    # Calculate errors
    abs_error = np.mean(np.abs(original - dequantized))
    rel_error = np.mean(np.abs((original - dequantized) / (original + 1e-8)))
    max_error = np.max(np.abs(original - dequantized))
    
    # Check if quantization is acceptable
    is_valid = abs_error < tolerance
    
    return {
        'is_valid': is_valid,
        'abs_error': abs_error,
        'rel_error': rel_error,
        'max_error': max_error,
        'tolerance': tolerance
    }


def analyze_weight_distribution(weights, name="weights"):
    """
    Analyze weight distribution for better quantization strategy selection
    """
    weights_flat = weights.flatten()
    
    stats = {
        'name': name,
        'min': np.min(weights_flat),
        'max': np.max(weights_flat),
        'mean': np.mean(weights_flat),
        'std': np.std(weights_flat),
        'median': np.median(weights_flat),
        'q25': np.percentile(weights_flat, 25),
        'q75': np.percentile(weights_flat, 75),
        'zeros': np.sum(weights_flat == 0),
        'total_elements': len(weights_flat)
    }
    
    # Recommend quantization strategy
    abs_max = max(abs(stats['min']), abs(stats['max']))
    range_val = stats['max'] - stats['min']
    
    # If the distribution is roughly symmetric around zero, use symmetric
    if abs(stats['mean']) < 0.1 * stats['std'] and abs(stats['min']) / abs_max > 0.7:
        recommended_strategy = 'symmetric'
    else:
        recommended_strategy = 'asymmetric'
    
    stats['recommended_strategy'] = recommended_strategy
    stats['symmetry_score'] = abs(stats['min']) / abs_max if abs_max > 0 else 1.0
    
    return stats