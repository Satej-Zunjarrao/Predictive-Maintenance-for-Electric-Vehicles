"""
utils.py

This script contains utility functions and helpers used across different modules of the Predictive Maintenance System.
These functions ensure code reusability and cleaner implementation.

Author: Satej
"""

import os
import json
import numpy as np
import pandas as pd

def load_json(file_path):
    """
    Load data from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON data.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        print(f"JSON file loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Failed to parse JSON file: {file_path}")
        return None

def save_json(data, file_path):
    """
    Save data to a JSON file.

    Args:
        data (dict): Data to save.
        file_path (str): Path to save the JSON file.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"JSON file saved successfully to {file_path}")

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics for model performance.

    Args:
        y_true (np.array): True target values.
        y_pred (np.array): Predicted target values.

    Returns:
        dict: Calculated metrics (Mean Absolute Error and R-squared).
    """
    mae = np.mean(np.abs(y_true - y_pred))
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    r2 = 1 - (ss_residual / ss_total)
    print(f"Metrics calculated: MAE={mae:.4f}, R²={r2:.4f}")
    return {"MAE": mae, "R²": r2}

def ensure_dir_exists(directory):
    """
    Ensure a directory exists; create it if it doesn't.

    Args:
        directory (str): Directory path.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")
    else:
        print(f"Directory already exists: {directory}")

def load_csv(file_path):
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"CSV file loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def save_csv(df, file_path):
    """
    Save a pandas DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        file_path (str): Path to save the CSV file.
    """
    df.to_csv(file_path, index=False)
    print(f"DataFrame saved successfully to {file_path}")
