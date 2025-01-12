"""
data_preprocessing.py

This script handles data collection, cleaning, and preprocessing steps such as handling missing values, removing noise,
and aligning time-series data for the Predictive Maintenance System for Electric Vehicle batteries.

Author: Satej
"""

import pandas as pd
import numpy as np

# Configuration for input and output paths
INPUT_FILE = "data/battery_telemetry.csv"  # Replace with the actual file path
OUTPUT_FILE = "processed_data/cleaned_telemetry.csv"

def load_data(file_path):
    """
    Load telemetry data from a CSV file.

    Args:
        file_path (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: Loaded telemetry data.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def clean_data(df):
    """
    Clean telemetry data by handling missing values and removing duplicates.

    Args:
        df (pd.DataFrame): Raw telemetry data.

    Returns:
        pd.DataFrame: Cleaned telemetry data.
    """
    # Drop duplicate rows
    df = df.drop_duplicates()
    print("Duplicate rows removed.")

    # Fill missing numerical values with column mean
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mean(), inplace=True)
            print(f"Missing values in column '{col}' filled with mean.")

    return df

def align_time_series(df, time_col):
    """
    Align time-series data to ensure uniform timestamps across sensors.

    Args:
        df (pd.DataFrame): Telemetry data.
        time_col (str): Name of the timestamp column.

    Returns:
        pd.DataFrame: Time-aligned telemetry data.
    """
    # Convert the timestamp column to datetime
    df[time_col] = pd.to_datetime(df[time_col])

    # Resample data to 1-minute intervals
    df = df.set_index(time_col).resample('1T').mean().reset_index()
    print("Time-series data aligned to 1-minute intervals.")

    return df

def save_processed_data(df, output_path):
    """
    Save the processed data to a CSV file.

    Args:
        df (pd.DataFrame): Processed data.
        output_path (str): Path to save the processed CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

def main():
    # Load the raw telemetry data
    raw_data = load_data(INPUT_FILE)
    if raw_data is None:
        return

    # Clean the data
    cleaned_data = clean_data(raw_data)

    # Align time-series data
    processed_data = align_time_series(cleaned_data, time_col='timestamp')

    # Save the processed data
    save_processed_data(processed_data, OUTPUT_FILE)

if __name__ == "__main__":
    main()
