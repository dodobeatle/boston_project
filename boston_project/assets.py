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
    return Output(
        df, 
        metadata={"num_samples": df.shape[0]}
    )

@multi_asset(
    #name="preprocess_data",
    description="Preprocess the Boston Housing Data",
    group_name="data_preprocessing",
    ins={
        "boston_housing_data":AssetIn()
    },
    outs={
        "preprocessed_data":AssetOut()
    },
   required_resource_keys={"mlflow"}
)
def preprocess_data(context: AssetExecutionContext, boston_housing_data: pd.DataFrame):
    mlflow = context.resources.mlflow
    
    preprocessed_data = boston_housing_data.copy()

    deleted_columns = ["age", "indus", "zn", "tax"]
    preprocessed_data = preprocessed_data.drop(columns=deleted_columns)

    mlflow.log_param("num_samples", preprocessed_data.shape[0])
    mlflow.log_param("num_features", preprocessed_data.shape[1])
    return preprocessed_data    

@multi_asset(
    group_name="data_preprocessing",
    description="Split the Boston Housing Data into training and testing sets",
    ins={"preprocessed_data":AssetIn()},
    outs={
        "X_train":AssetOut(), 
        "y_train":AssetOut(), 
        "X_test":AssetOut(), 
        "y_test":AssetOut()
    },
    required_resource_keys={"mlflow"}
)
def split_data(context: AssetExecutionContext, preprocessed_data: pd.DataFrame):
    # Separar características (X) y variable objetivo (y)
    X = preprocessed_data.drop('medv', axis=1)
    y = preprocessed_data['medv']

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

    mlflow.statsmodels.log_model(lm_model, "linear_regression_model")
    return lm_model

@asset(
    group_name="model_evaluation",
    description="Evaluate the Linear Regression Model",
    ins={
        "lm_model":AssetIn(),
        "X_test":AssetIn(), 
        "y_test":AssetIn()
    },
    required_resource_keys={"mlflow"}
)
def evaluate_linear_regression(context: AssetExecutionContext, lm_model, X_test: pd.DataFrame, y_test: pd.Series):
    mlflow = context.resources.mlflow
    y_pred = lm_model.predict(sm.add_constant(X_test))    
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

