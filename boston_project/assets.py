from dagster import asset, Output, AssetIn, AssetOut, multi_asset, AssetExecutionContext, AssetKey, file_relative_path
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm  
from .utils.eval_metrics import eval_metrics    
from .utils.fig_helper import plot_predictions, plot_correlation_matrix
from dagstermill import define_dagstermill_asset

@asset(
    description="Load the Boston Housing Data",
    group_name="data_ingestion",    
    ins={
        "boston_housing_data":AssetIn(key=AssetKey("boston_housing"),
        input_manager_key="postgres_io_manager")
    }
)
def boston_housing_data(context: AssetExecutionContext, boston_housing_data: pd.DataFrame) -> Output[pd.DataFrame]:
    #df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
    df = boston_housing_data.copy()
    return Output(
        df, 
        metadata={"num_samples": df.shape[0]}
    )

@multi_asset(
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

    deleted_columns = ["age", "indus", "zn", "tax" ,"rad", "crim", "chas"]
    preprocessed_data = preprocessed_data.drop(columns=deleted_columns)  
    
    # Log the figure to MLflow
    fig_path = plot_correlation_matrix(preprocessed_data)
    context.resources.mlflow.log_artifact(fig_path)

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
    ins={
        "X_train":AssetIn(),
        "y_train":AssetIn()
    },
    outs={
        "lm_model":AssetOut()
    },
    compute_kind="python",
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
    import matplotlib.pyplot as plt
    y_pred = lm_model.predict(sm.add_constant(X_test))   
    mlflow = context.resources.mlflow

    fig_name = plot_predictions(y_test, y_pred)
    mlflow.log_artifact(fig_name)

    metrics = eval_metrics(y_test, y_pred)
    mlflow.log_metrics(metrics)
    return metrics


boston_eda_jupyter_notebook = define_dagstermill_asset(
    name="boston_jupyter_notebook",
    notebook_path=file_relative_path(__file__, "notebooks/boston.ipynb"),
    group_name="data_ingestion",    
    ins={
        "boston":AssetIn(key=AssetKey("boston_housing_data"))
    }
)