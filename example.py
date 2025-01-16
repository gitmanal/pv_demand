import pandas as pd
import numpy as np
from utils import *
from sklearn.ensemble import RandomForestRegressor
import mlflow

# Example usage
if __name__ == "__main__":
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    values = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 0.1, len(dates))
    df = pd.DataFrame({'date': dates, 'value': values})
    
    # Split data
    train, valid, test = train_test_split_timeseries(df)
    
    # Create sequences
    X_train, y_train = create_sequences(train['value'], seq_length=7)
    X_test, y_test = create_sequences(test['value'], seq_length=7)
    
    # Train model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate and log results
    metrics = evaluate_timeseries_model(y_test, y_pred, "example_run")
    
    # Log experiment
    params = {
        'seq_length': 7,
        'n_estimators': model.n_estimators
    }
    
    log_experiment(
        model=model,
        params=params,
        metrics=metrics
    )

