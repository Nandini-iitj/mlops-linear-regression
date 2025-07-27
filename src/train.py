"""
Train Linear Regression model on California housing dataset.
"""
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from utils import load_data, save_model


def train_model():
    """
    Train linear regression model and save results.
    
    Returns:
        LinearRegression: Trained model
    """
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    
    # Initialize and train model
    print("\nTraining Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Calculate performance metrics
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    
    # Display results
    print(f"\n--- Training Results ---")
    print(f"Train R² Score: {train_r2:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"Train MSE Loss: {train_mse:.4f}")
    print(f"Test MSE Loss: {test_mse:.4f}")
    
    # Save model and test data
    os.makedirs('models', exist_ok=True)
    save_model(model, 'models/model.joblib')
    save_model((X_test, y_test), 'models/test_data.joblib')
    
    print("\nModel training completed successfully!")
    return model


if __name__ == "__main__":
    train_model()