"""
dashboard_visualization.py

This script creates a dashboard to visualize battery health metrics, Remaining Useful Life (RUL) predictions,
and maintenance schedules for fleet management. Uses Dash for web-based visualization.

Author: Satej
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go

# Configuration for file paths
INPUT_FILE = "processed_data/engineered_features.csv"
RUL_PREDICTIONS_FILE = "processed_data/rul_predictions.csv"

# Load data
df = pd.read_csv(INPUT_FILE)
rul_predictions = pd.read_csv(RUL_PREDICTIONS_FILE)

# Initialize Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("EV Battery Health Dashboard", style={"textAlign": "center"}),

    dcc.Graph(id='soc-over-time'),
    dcc.Graph(id='rul-distribution'),

    html.Label("Select Feature for Analysis:"),
    dcc.Dropdown(
        id='feature-dropdown',
        options=[{'label': col, 'value': col} for col in df.columns if col != 'timestamp'],
        value='state_of_charge',
        style={"width": "50%"}
    ),
    dcc.Graph(id='feature-visualization')
])

# Callbacks for dynamic updates
@app.callback(
    Output('soc-over-time', 'figure'),
    Input('feature-dropdown', 'value')
)
def update_soc_graph(selected_feature):
    """
    Updates the SOC over time graph based on user selection.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['state_of_charge'], mode='lines', name='SOC'))
    fig.update_layout(
        title="State of Charge Over Time",
        xaxis_title="Time",
        yaxis_title="State of Charge (%)"
    )
    return fig

@app.callback(
    Output('rul-distribution', 'figure'),
    Input('feature-dropdown', 'value')
)
def update_rul_distribution(selected_feature):
    """
    Updates the RUL distribution graph.
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=rul_predictions['rul'], nbinsx=30, name='RUL Distribution'))
    fig.update_layout(
        title="Distribution of RUL Predictions",
        xaxis_title="Remaining Useful Life (RUL)",
        yaxis_title="Frequency"
    )
    return fig

@app.callback(
    Output('feature-visualization', 'figure'),
    Input('feature-dropdown', 'value')
)
def update_feature_graph(selected_feature):
    """
    Visualizes the selected feature over time.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df[selected_feature], mode='lines', name=selected_feature))
    fig.update_layout(
        title=f"{selected_feature} Over Time",
        xaxis_title="Time",
        yaxis_title=selected_feature
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
