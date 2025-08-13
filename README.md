# Studying and Predicting Energy demand on Photovoltaic Panels

This project provides a template for time series regression using MLflow for experiment tracking.

## Setup
1. Run the setup script: `./setup.sh`
2. Activate virtual environment: `source venv/bin/activate`
3. Start MLflow UI: `mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns`
4. Access the UI at http://localhost:5000

## Project Structure
- `data/`: Store your datasets
- `models/`: Saved models
- `notebooks/`: Jupyter notebooks
- `utils.py`: Utility functions for time series analysis
- `example.py`: Example usage of utilities

## Usage
See `example.py` for sample code demonstrating how to:
- Create sequences from time series data
- Split data into train/validation/test sets
- Train and evaluate models
- Log experiments with MLflow


<!-- Test change for collaborator badge -->

