from dagster import asset, Output, AssetIn, AssetOut, multi_asset, AssetExecutionContext
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import mlflow   

@asset(
    description="Load the Boston Housing Data",
    group_name="data_ingestion",
    required_resource_keys={"mlflow"}
)
def boston_housing_data(context: AssetExecutionContext) -> Output[pd.DataFrame]:
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
    mlflow = context.resources.mlflow
    mlflow.log_param("num_samples", df.shape[0])
    mlflow.log_param("num_features", df.shape[1])
    return Output(
        df, 
        metadata={"num_samples": df.shape[0]}
    )

# @asset(
#     description="Summary of the Boston Housing Data",
#     ins={"boston_housing_data":AssetIn( )}
# )
# def boston_housing_data_summary(boston_housing_data: pd.DataFrame) -> pd.DataFrame:
#     return boston_housing_data.describe()


@multi_asset(
    group_name="data_preprocessing",
    description="Split the Boston Housing Data into training and testing sets",
    ins={"boston_housing_data":AssetIn()},
    outs={
        "X_train":AssetOut(), 
        "y_train":AssetOut(), 
        "X_test":AssetOut(), 
        "y_test":AssetOut()
    },
    required_resource_keys={"mlflow"}
)
def split_data(context: AssetExecutionContext, boston_housing_data: pd.DataFrame):
    # Separar características (X) y variable objetivo (y)
    X = boston_housing_data.drop('medv', axis=1)
    y = boston_housing_data['medv']

    split_params = {
        "test_size": 0.2,
        "random_state": 42
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_params)
    
    mlflow = context.resources.mlflow
    mlflow.log_params(split_params)
    
    return X_train, y_train, X_test, y_test


@multi_asset(
    group_name="model_training",
    description="Train a Linear Regression Model",
    ins={"X_train":AssetIn(), "y_train":AssetIn()},
    outs={"lm_model":AssetOut()},
    required_resource_keys={"mlflow"}
)
def train_linear_regression(context: AssetExecutionContext, X_train: pd.DataFrame, y_train: pd.Series):
    mlflow = context.resources.mlflow
    mlflow.statsmodels.autolog()
    X_train = sm.add_constant(X_train)
    lm_model = sm.OLS(y_train, X_train).fit()
    return lm_model

@asset(
    group_name="model_evaluation",
    description="Evaluate the Linear Regression Model",
    ins={"lm_model":AssetIn(), "X_test":AssetIn(), "y_test":AssetIn()},
    required_resource_keys={"mlflow"}
)
def evaluate_linear_regression(context: AssetExecutionContext, lm_model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = lm_model.predict(sm.add_constant(X_test))    
    mlflow = context.resources.mlflow
    metrics = eval_metrics(y_test, y_pred)
    mlflow.log_metrics(metrics)
    return metrics
    


def eval_metrics(y_test, y_pred):
    """
    Calcula múltiples métricas de evaluación para un modelo de regresión
    
    Args:
        y_test: Valores reales
        y_pred: Valores predichos
        
    Returns:
        dict: Diccionario con las métricas MSE, RMSE y MAE
    """
    mse = ((y_pred - y_test) ** 2).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(y_pred - y_test).mean()
    
    metrics = {
        'mse': mse,
        'rmse': rmse, 
        'mae': mae
    }
    
    return metrics

