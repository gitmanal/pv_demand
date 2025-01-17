import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, List, Dict
import mlflow

def create_sequences(data: pd.Series, 
                    seq_length: int, 
                    horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction.
    
    Args:
        data: Time series data
        seq_length: Length of input sequence
        horizon: Prediction horizon
    
    Returns:
        Tuple of input sequences and target values
    """
    X, y = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length:i + seq_length + horizon])
    return np.array(X), np.array(y)

def evaluate_timeseries_model(y_true: np.ndarray, 
                            y_pred: np.ndarray, 
                            run_name: str = None) -> Dict:
    """
    Evaluate time series model performance.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        run_name: MLflow run name
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    if run_name:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_metrics(metrics)
    
    return metrics

def train_test_split_timeseries(data: pd.DataFrame,
                              test_ratio: float = 0.2,
                              valid_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train, validation, and test sets.
    
    Args:
        data: Time series dataframe
        test_ratio: Ratio of test set
        valid_ratio: Ratio of validation set
    
    Returns:
        Tuple of train, validation, and test dataframes
    """
    n = len(data)
    test_size = int(n * test_ratio)
    valid_size = int(n * valid_ratio)
    
    test = data[-test_size:]
    valid = data[-(test_size + valid_size):-test_size]
    train = data[:-(test_size + valid_size)]
    
    return train, valid, test

def log_experiment(model, 
                  params: Dict,
                  metrics: Dict,
                  artifacts: Dict = None):
    """
    Log experiment details to MLflow.
    
    Args:
        model: Trained model
        params: Model parameters
        metrics: Model metrics
        artifacts: Additional artifacts to log
    """
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log additional artifacts
        if artifacts:
            for name, path in artifacts.items():
                mlflow.log_artifact(path, name)

