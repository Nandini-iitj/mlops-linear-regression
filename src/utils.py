"""
Utility functions for MLOps pipeline.
"""
import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def load_data(test_size=0.2, random_state=42):
    """
    Load and split California housing dataset.
    
    Args:
        test_size (float): Proportion of test data
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("Loading California housing dataset...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def save_model(model, filepath):
    """Save model using joblib."""
    joblib.dump(model, filepath)
    print(f"Saved: {filepath}")


def load_model(filepath):
    """Load model using joblib."""
    return joblib.load(filepath)


def quantize_value(value, min_val, max_val):
    """
    Convert float value to 8-bit unsigned integer.
    
    Args:
        value: Float value to quantize
        min_val: Minimum value for scaling
        max_val: Maximum value for scaling
        
    Returns:
        numpy.uint8: Quantized value
    """
    scaled = (value - min_val) / (max_val - min_val) * 255
    return np.round(scaled).astype(np.uint8)


def dequantize_value(quantized, min_val, max_val):
    """
    Convert 8-bit unsigned integer back to float.
    
    Args:
        quantized: Quantized value
        min_val: Minimum value for scaling
        max_val: Maximum value for scaling
        
    Returns:
        numpy.float32: Dequantized value
    """
    return (quantized.astype(np.float32) / 255) * (max_val - min_val) + min_val