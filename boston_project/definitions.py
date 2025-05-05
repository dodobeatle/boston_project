from dagster import Definitions, load_assets_from_modules, define_asset_job, AssetSelection

from boston_project import assets  # noqa: TID252
from dagster_mlflow import mlflow_tracking
from boston_project.resources.airbyte import airbyte_connection
from boston_project.resources import dbt_resource, postgres_io_manager
from boston_project.resources import dbt
from dagstermill import ConfigurableLocalOutputNotebookIOManager
#------------------------------- Assets -------------------------------------
#airbyte_assets = load_assets_from_modules([airbyte_connection])
all_assets = load_assets_from_modules([assets])

dbt_assets = load_assets_from_modules([dbt], group_name="raw_data_ingestion")

#------------------------------- Jobs ---------------------------------------
airbyte_sync_job = define_asset_job(name="airbyte_sync_job",  selection=AssetSelection.groups("raw_data_ingestion"))

dbt_sync_job = define_asset_job(name="dbt_sync_job", selection=AssetSelection.groups("raw_data_ingestion"))

preprocessing_trained_evaluation_job = define_asset_job(name = "preprocessing_trained_evaluation_job", selection = AssetSelection.groups("data_preprocessing","model_training", "model_evaluation") )

trained_evaluation_job = define_asset_job(name = "trained_evaluation_job", selection = AssetSelection.groups("model_training", "model_evaluation") )

all_sync_job = define_asset_job(name = "all_sync_job", selection = "*" )
#------------------------------- Schedules ----------------------------------

#------------------------------- Definitions --------------------------------
defs = Definitions(
    assets=[airbyte_connection, *dbt_assets, *all_assets],
    jobs=[airbyte_sync_job, dbt_sync_job, preprocessing_trained_evaluation_job, trained_evaluation_job, all_sync_job],
    resources={
        "dbt": dbt_resource,
        "output_notebook_io_manager": ConfigurableLocalOutputNotebookIOManager(),
        "postgres_io_manager": postgres_io_manager.configured({
            "connection_string": "env:POSTGRES_CONNECTION_STRING",
            "schema": "target"}),
        "mlflow": mlflow_tracking.configured({
            "mlflow_tracking_uri": {"env":"MLFLOW_TRACKING_URI"}, #config_parameters["mlflow"]["mlflow_tracking_uri"],
            "experiment_name": {"env": "MLFLOW_EXPERIMENT_NAME"} #config_parameters["mlflow"]["experiment_name"]
        })
    }   
)
