"""
feature_engineering.py

This script performs feature engineering on the cleaned telemetry data. It includes generating features like depth
of discharge (DoD), charge/discharge rates, cumulative energy throughput, rolling averages, and lagged metrics.

Author: Satej
"""

import pandas as pd

# Configuration for file paths
INPUT_FILE = "processed_data/cleaned_telemetry.csv"
OUTPUT_FILE = "processed_data/engineered_features.csv"

def calculate_depth_of_discharge(df):
    """
    Calculate the Depth of Discharge (DoD) based on the state of charge (SOC).

    Args:
        df (pd.DataFrame): Telemetry data.

    Returns:
        pd.DataFrame: Data with an additional 'depth_of_discharge' column.
    """
    df['depth_of_discharge'] = 100 - df['state_of_charge']
    print("Depth of Discharge (DoD) calculated.")
    return df

def calculate_charge_discharge_rates(df):
    """
    Calculate the charge and discharge rates.

    Args:
        df (pd.DataFrame): Telemetry data.

    Returns:
        pd.DataFrame: Data with additional 'charge_rate' and 'discharge_rate' columns.
    """
    df['charge_rate'] = df['current'].apply(lambda x: x if x > 0 else 0)
    df['discharge_rate'] = df['current'].apply(lambda x: abs(x) if x < 0 else 0)
    print("Charge and discharge rates calculated.")
    return df

def cumulative_energy_throughput(df):
    """
    Calculate the cumulative energy throughput over time.

    Args:
        df (pd.DataFrame): Telemetry data.

    Returns:
        pd.DataFrame: Data with an additional 'cumulative_energy' column.
    """
    df['cumulative_energy'] = (df['voltage'] * df['current']).cumsum()
    print("Cumulative energy throughput calculated.")
    return df

def add_rolling_features(df, column, window_size):
    """
    Add rolling average features to capture temporal patterns.

    Args:
        df (pd.DataFrame): Telemetry data.
        column (str): Column name for which to calculate the rolling average.
        window_size (int): Window size for the rolling average.

    Returns:
        pd.DataFrame: Data with an additional rolling average column.
    """
    rolling_col_name = f"{column}_rolling_avg_{window_size}"
    df[rolling_col_name] = df[column].rolling(window=window_size).mean()
    print(f"Rolling average feature '{rolling_col_name}' created.")
    return df

def add_lagged_features(df, column, lags):
    """
    Add lagged features to model temporal dependencies.

    Args:
        df (pd.DataFrame): Telemetry data.
        column (str): Column name for which to calculate lagged features.
        lags (list): List of lag values.

    Returns:
        pd.DataFrame: Data with additional lagged feature columns.
    """
    for lag in lags:
        lag_col_name = f"{column}_lag_{lag}"
        df[lag_col_name] = df[column].shift(lag)
        print(f"Lagged feature '{lag_col_name}' created.")
    return df

def main():
    # Load the cleaned telemetry data
    df = pd.read_csv(INPUT_FILE)
    print(f"Data loaded from {INPUT_FILE}")

    # Feature engineering steps
    df = calculate_depth_of_discharge(df)
    df = calculate_charge_discharge_rates(df)
    df = cumulative_energy_throughput(df)

    # Add rolling average and lagged features
    df = add_rolling_features(df, column='temperature', window_size=5)
    df = add_lagged_features(df, column='state_of_charge', lags=[1, 2, 3])

    # Save the engineered data
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Engineered features saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
