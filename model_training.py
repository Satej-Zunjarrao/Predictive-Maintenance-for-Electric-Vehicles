"""
model_training.py

This script trains predictive models to estimate the remaining useful life (RUL) of EV batteries. It includes
a regression model using Scikit-learn and an LSTM model using TensorFlow.

Author: Satej
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Configuration for file paths
INPUT_FILE = "processed_data/engineered_features.csv"
MODEL_DIR = "trained_models/"

def train_regression_model(X_train, y_train):
    """
    Train a Random Forest regression model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.

    Returns:
        RandomForestRegressor: Trained model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Regression model trained.")
    return model

def train_lstm_model(X_train, y_train, input_shape):
    """
    Train an LSTM model.

    Args:
        X_train (np.array): Training features reshaped for LSTM.
        y_train (np.array): Training target variable.
        input_shape (tuple): Shape of the input for the LSTM model.

    Returns:
        Sequential: Trained LSTM model.
    """
    model = Sequential([
        LSTM(50, input_shape=input_shape, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    print("LSTM model trained.")
    return model

def evaluate_model(model, X_test, y_test, model_type="Regression"):
    """
    Evaluate the model's performance on the test set.

    Args:
        model: Trained model (RandomForestRegressor or Sequential).
        X_test (pd.DataFrame or np.array): Test features.
        y_test (pd.Series or np.array): Test target variable.
        model_type (str): Type of model ("Regression" or "LSTM").

    Returns:
        dict: Performance metrics.
    """
    if model_type == "Regression":
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test).flatten()

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_type} Model Performance - MAE: {mae:.4f}, R²: {r2:.4f}")
    return {"MAE": mae, "R²": r2}

def main():
    # Load the engineered data
    df = pd.read_csv(INPUT_FILE)
    print(f"Data loaded from {INPUT_FILE}")

    # Define features and target variable
    X = df.drop(columns=['remaining_useful_life'])
    y = df['remaining_useful_life']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Train-test split completed.")

    # Train the regression model
    regression_model = train_regression_model(X_train, y_train)
    evaluate_model(regression_model, X_test, y_test, model_type="Regression")

    # Prepare data for LSTM (reshape to 3D for time-series)
    X_train_lstm = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Train the LSTM model
    lstm_model = train_lstm_model(X_train_lstm, y_train.values, input_shape=(X_train.shape[1], 1))
    evaluate_model(lstm_model, X_test_lstm, y_test.values, model_type="LSTM")

if __name__ == "__main__":
    main()
