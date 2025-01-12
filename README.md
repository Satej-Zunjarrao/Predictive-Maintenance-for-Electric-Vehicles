# Predictive Maintenance System for EV Batteries

Built a time-series forecasting system to estimate the Remaining Useful Life (RUL) of electric vehicle batteries.

---

## Overview

The **Predictive Maintenance System for EV Batteries** is a Python-based solution designed to predict the Remaining Useful Life (RUL) of electric vehicle (EV) batteries and optimize maintenance schedules. By leveraging advanced machine learning techniques and time-series forecasting, this system reduces unexpected failures and enhances fleet reliability.

This project includes a modular pipeline for data preprocessing, exploratory analysis, feature engineering, model training, deployment, and visualization.

---

## Key Features

- **Data Preprocessing**: Cleans and standardizes telemetry data, aligning time-series data from multiple sensors.
- **Exploratory Data Analysis (EDA)**: Visualizes battery health metrics and trends in telemetry data.
- **Feature Engineering**: Generates features like depth of discharge, charge/discharge rates, and lagged metrics.
- **Model Training**: Trains regression and LSTM models to predict RUL with high accuracy.
- **Model Deployment**: Deploys trained models via Flask API for real-time predictions.
- **Visualization**: Creates dashboards to display battery health metrics, RUL predictions, and maintenance schedules.

---

## Directory Structure

```plaintext
project/
│
├── data_preprocessing.py       # Handles data cleaning and time-series alignment
├── eda_visualization.py        # Generates visualizations and insights from telemetry data
├── feature_engineering.py      # Creates derived features for improved model performance
├── model_training.py           # Trains and evaluates regression and LSTM models
├── model_deployment.py         # Deploys the model as a Flask API for real-time predictions
├── dashboard_visualization.py  # Builds a dashboard for fleet managers
├── utils.py                    # Provides helper functions for common tasks
├── config.py                   # Centralized configuration for paths and parameters
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
```

## Modules

### 1. `data_preprocessing.py`
- Loads raw telemetry data and preprocesses it.
- Handles missing values, removes duplicates, and aligns time-series data.

### 2. `eda_visualization.py`
- Performs exploratory data analysis on telemetry data.
- Generates visualizations such as SOC trends and temperature distributions.

### 3. `feature_engineering.py`
- Creates features like depth of discharge (DoD), charge/discharge rates, and lagged metrics.
- Enhances data quality for improved model performance.

### 4. `model_training.py`
- Trains machine learning models for RUL prediction.
- Includes Random Forest regression and LSTM models for short-term and long-term dependencies.

### 5. `model_deployment.py`
- Deploys the trained models as a Flask API.
- Enables real-time predictions for fleet management systems.

### 6. `dashboard_visualization.py`
- Builds an interactive dashboard using Dash.
- Visualizes battery health, RUL predictions, and maintenance schedules.

### 7. `utils.py`
- Provides reusable utility functions for logging, metrics, and directory management.

### 8. `config.py`
- Centralized configuration file for paths, parameters, and settings.
- Simplifies updates to project configurations.

### 9. `requirements.txt`
- Lists all dependencies required for the project.
- Ensures a seamless setup with compatible versions.

---

## Contact

For queries or collaboration, feel free to reach out:

- **Name**: Satej Zunjarrao  
- **Email**: zsatej1028@gmail.com
