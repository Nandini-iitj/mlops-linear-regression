"""
Model prediction script for verification.
"""
import os
from utils import load_model


def run_predictions():
    """
    Load trained model and run predictions on test data.
    """
    # Check if required files exist
    if not os.path.exists('models/model.joblib'):
        raise FileNotFoundError("Model not found. Please run train.py first.")
    
    if not os.path.exists('models/test_data.joblib'):
        raise FileNotFoundError("Test data not found. Please run train.py first.")
    
    print("Loading trained model and test data...")
    model = load_model('models/model.joblib')
    X_test, y_test = load_model('models/test_data.joblib')
    
    print("Running model predictions...")
    predictions = model.predict(X_test)
    r2_score = model.score(X_test, y_test)
    
    # Display results
    print(f"\n--- Model Performance ---")
    print(f"R² Score on test set: {r2_score:.4f}")
    print(f"Total test samples: {len(y_test)}")
    
    print(f"\n--- Sample Outputs (first 10) ---")
    print("Actual vs Predicted:")
    for i in range(min(10, len(y_test))):
        print(f"Sample {i+1:2d}: Actual={y_test[i]:.3f}, Predicted={predictions[i]:.3f}")
    
    # Model information
    print(f"\n--- Model Information ---")
    print(f"Number of features: {len(model.coef_)}")
    print(f"Model intercept: {model.intercept_:.6f}")
    print(f"First 5 coefficients: {model.coef_[:5]}")
    
    print("\nPrediction verification completed successfully!")
    return predictions, r2_score


if __name__ == "__main__":
    run_predictions()