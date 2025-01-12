"""
model_deployment.py

This script deploys the trained model using Flask to provide real-time predictions of the Remaining Useful Life (RUL)
for EV batteries based on telemetry data. The API can be integrated with external systems for monitoring and alerts.

Author: Satej
"""

from flask import Flask, request, jsonify
import pandas as pd
import joblib  # For loading the regression model
import tensorflow as tf  # For loading the LSTM model

# Configuration for model paths
REGRESSION_MODEL_PATH = "trained_models/regression_model.pkl"
LSTM_MODEL_PATH = "trained_models/lstm_model.h5"

# Load the trained models
regression_model = joblib.load(REGRESSION_MODEL_PATH)
lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_rul():
    """
    API endpoint to predict the Remaining Useful Life (RUL) of an EV battery.

    Expects JSON input with telemetry data.

    Example Input:
    {
        "telemetry": {
            "feature1": value1,
            "feature2": value2,
            ...
        }
    }

    Returns:
        JSON response with the predicted RUL.
    """
    try:
        # Parse input data
        input_data = request.json.get("telemetry", {})
        if not input_data:
            return jsonify({"error": "No telemetry data provided"}), 400

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Predict using regression model
        regression_prediction = regression_model.predict(input_df)[0]

        # Prepare data for LSTM (reshaping to 3D)
        lstm_input = input_df.values.reshape((1, input_df.shape[1], 1))
        lstm_prediction = lstm_model.predict(lstm_input).flatten()[0]

        # Combine predictions (example: simple average)
        final_prediction = (regression_prediction + lstm_prediction) / 2

        return jsonify({
            "rul_prediction": final_prediction,
            "details": {
                "regression_model_prediction": regression_prediction,
                "lstm_model_prediction": lstm_prediction
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
