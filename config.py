"""
config.py

This script centralizes configuration variables for the Predictive Maintenance System. It allows easy updates
to file paths, parameters, and settings across all modules.

Author: Satej
"""

# File Paths
RAW_DATA_FILE = "data/battery_telemetry.csv"
CLEANED_DATA_FILE = "processed_data/cleaned_telemetry.csv"
FEATURE_ENGINEERED_FILE = "processed_data/engineered_features.csv"
RUL_PREDICTIONS_FILE = "processed_data/rul_predictions.csv"
REGRESSION_MODEL_PATH = "trained_models/regression_model.pkl"
LSTM_MODEL_PATH = "trained_models/lstm_model.h5"
EDA_OUTPUT_DIR = "eda_plots/"

# Model Training Parameters
TEST_SIZE = 0.2  # Proportion of data to use for testing
RANDOM_STATE = 42  # Random state for reproducibility

# LSTM Hyperparameters
LSTM_EPOCHS = 10
LSTM_BATCH_SIZE = 32
LSTM_UNITS = 50

# Feature Engineering Parameters
ROLLING_WINDOW_SIZE = 5  # Window size for rolling averages
LAGS = [1, 2, 3]  # Lag intervals for lagged features

# Flask Deployment
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True

# Dashboard Configuration
DASHBOARD_HOST = '127.0.0.1'
DASHBOARD_PORT = 8050
DASHBOARD_DEBUG = True
