from dagster import Definitions, load_assets_from_modules, define_asset_job, AssetSelection

from boston_project import assets  # noqa: TID252
from dagster_mlflow import mlflow_tracking
from boston_project.airbyte import airbyte_connection

#------------------------------- Assets -------------------------------------
#airbyte_assets = load_assets_from_modules([airbyte_connection])
all_assets = load_assets_from_modules([assets])

#------------------------------- Jobs ---------------------------------------
airbyte_sync_job = define_asset_job(name="airbyte_sync_job",  selection=AssetSelection.groups("raw_data_ingestion"))

preprocessing_trained_evaluation_job = define_asset_job(name = "preprocessing_trained_evaluation_job", selection = AssetSelection.groups("data_preprocessing","model_training", "model_evaluation") )

trained_evaluation_job = define_asset_job(name = "trained_evaluation_job", selection = AssetSelection.groups("model_training", "model_evaluation") )

all_sync_job = define_asset_job(name = "all_sync_job", selection = "*" )
#------------------------------- Schedules ----------------------------------

#------------------------------- Definitions --------------------------------
defs = Definitions(
    assets=[airbyte_connection, *all_assets],
    jobs=[all_sync_job, preprocessing_trained_evaluation_job, trained_evaluation_job],
    resources={
        "mlflow": mlflow_tracking.configured({
            "mlflow_tracking_uri": {"env":"MLFLOW_TRACKING_URI"}, #config_parameters["mlflow"]["mlflow_tracking_uri"],
            "experiment_name": {"env": "MLFLOW_EXPERIMENT_NAME"} #config_parameters["mlflow"]["experiment_name"]
        })
    }   
)
