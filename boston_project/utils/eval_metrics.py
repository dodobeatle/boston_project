from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def eval_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }
    return metrics