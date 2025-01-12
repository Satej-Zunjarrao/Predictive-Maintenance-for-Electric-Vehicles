"""
eda_visualization.py

This script performs exploratory data analysis (EDA) and generates visualizations to identify trends and patterns in
the battery telemetry data for the Predictive Maintenance System.

Author: Satej
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration for file paths
INPUT_FILE = "processed_data/cleaned_telemetry.csv"
OUTPUT_DIR = "eda_plots/"

def load_data(file_path):
    """
    Load processed telemetry data from a CSV file.

    Args:
        file_path (str): Path to the processed CSV file.

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

def plot_battery_health(df):
    """
    Plot battery health metrics such as state of charge (SOC) over time.

    Args:
        df (pd.DataFrame): Telemetry data.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['state_of_charge'], label="State of Charge (SOC)")
    plt.xlabel("Time")
    plt.ylabel("State of Charge (%)")
    plt.title("Battery State of Charge Over Time")
    plt.legend()
    plt.grid()
    plt.savefig(f"{OUTPUT_DIR}soc_over_time.png")
    print("SOC over time plot saved.")
    plt.close()

def plot_temperature_distribution(df):
    """
    Plot the distribution of battery temperature.

    Args:
        df (pd.DataFrame): Telemetry data.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(df['temperature'], kde=True, bins=30, color='blue')
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")
    plt.title("Battery Temperature Distribution")
    plt.savefig(f"{OUTPUT_DIR}temperature_distribution.png")
    print("Temperature distribution plot saved.")
    plt.close()

def plot_correlations(df):
    """
    Plot a correlation heatmap for numerical features.

    Args:
        df (pd.DataFrame): Telemetry data.
    """
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.savefig(f"{OUTPUT_DIR}correlation_heatmap.png")
    print("Correlation heatmap saved.")
    plt.close()

def main():
    # Load the cleaned telemetry data
    data = load_data(INPUT_FILE)
    if data is None:
        return

    # Plot battery health metrics
    plot_battery_health(data)

    # Plot temperature distribution
    plot_temperature_distribution(data)

    # Plot feature correlations
    plot_correlations(data)

if __name__ == "__main__":
    main()
