"""
Unit tests for the training pipeline.
"""
import pytest
import os
import sys
import numpy as np

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sklearn.linear_model import LinearRegression
from utils import load_data, load_model
from train import train_model


class TestTrainingPipeline:
    """Test cases for the training pipeline."""
    
    def test_dataset_loading(self):
        """Test if dataset loads correctly."""
        X_train, X_test, y_train, y_test = load_data()
        
        # Check if data exists
        assert X_train is not None, "X_train should not be None"
        assert y_train is not None, "y_train should not be None"
        assert X_test is not None, "X_test should not be None"
        assert y_test is not None, "y_test should not be None"
        
        # Check data shapes
        assert X_train.shape[0] > 0, "Training set should have samples"
        assert X_test.shape[0] > 0, "Test set should have samples"
        assert X_train.shape[1] == 8, "California housing should have 8 features"
        assert len(y_train) == X_train.shape[0], "y_train length should match X_train"
        assert len(y_test) == X_test.shape[0], "y_test length should match X_test"
        
        # Check data types
        assert isinstance(X_train, np.ndarray), "X_train should be numpy array"
        assert isinstance(y_train, np.ndarray), "y_train should be numpy array"
        
        print("✓ Dataset loading test passed")
    
    def test_model_creation(self):
        """Validate model creation and LinearRegression instance."""
        model = LinearRegression()
        assert isinstance(model, LinearRegression), "Model should be LinearRegression instance"
        
        # Test model fitting
        X_train, X_test, y_train, y_test = load_data()
        model.fit(X_train, y_train)
        
        # Check if model has required attributes after fitting
        assert hasattr(model, 'coef_'), "Model should have coefficients"
        assert hasattr(model, 'intercept_'), "Model should have intercept"
        assert model.coef_ is not None, "Coefficients should not be None"
        assert model.intercept_ is not None, "Intercept should not be None"
        
        print("✓ Model creation test passed")
    
    def test_model_training(self):
        """Test if model was trained properly."""
        # Train the model
        model = train_model()
        
        # Check if model is trained (has coefficients)
        assert hasattr(model, 'coef_'), "Trained model should have coefficients"
        assert hasattr(model, 'intercept_'), "Trained model should have intercept"
        assert model.coef_ is not None, "Coefficients should exist"
        assert model.intercept_ is not None, "Intercept should exist"
        assert len(model.coef_) == 8, "Should have 8 coefficients for 8 features"
        
        # Check if model files are saved
        assert os.path.exists('models/model.joblib'), "Model file should be saved"
        assert os.path.exists('models/test_data.joblib'), "Test data should be saved"
        
        print("Model training test passed")
    
    def test_r2_score_threshold(self):
        """Ensure R² score exceeds minimum threshold."""
        # Load or train model if needed
        if not os.path.exists('models/model.joblib'):
            train_model()
        
        # Load model and test data
        model = load_model('models/model.joblib')
        X_test, y_test = load_model('models/test_data.joblib')
        
        # Calculate R² score
        r2_score = model.score(X_test, y_test)
        min_threshold = 0.5
        
        assert r2_score > min_threshold, f"R² score {r2_score:.4f} should be above {min_threshold}"
        
        print(f"R² threshold test passed (Score: {r2_score:.4f})")
    
    def test_model_file_persistence(self):
        """Test if model file can be saved and loaded correctly."""
        # Ensure model exists
        if not os.path.exists('models/model.joblib'):
            train_model()
        
        # Test file existence
        model_path = 'models/model.joblib'
        assert os.path.exists(model_path), "Model file should exist"
        
        # Test loading
        loaded_model = load_model(model_path)
        assert isinstance(loaded_model, LinearRegression), "Loaded model should be LinearRegression"
        assert hasattr(loaded_model, 'coef_'), "Loaded model should have coefficients"
        assert hasattr(loaded_model, 'intercept_'), "Loaded model should have intercept"
        
        print("Model file persistence test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])