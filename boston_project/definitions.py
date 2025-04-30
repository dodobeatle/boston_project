import yaml
from pathlib import Path
from dagster import Definitions, load_assets_from_modules, file_relative_path, define_asset_job

from boston_project import assets  # noqa: TID252
from dagster_mlflow import mlflow_tracking


# Load the cofigs.yml file (config parameters file)
config_file_path = file_relative_path(__file__, "configs.yml")
with open(config_file_path, "r") as file:
    config_parameters = yaml.safe_load(file)

#------------------------------- Assets -------------------------------------
all_assets = load_assets_from_modules([assets])

#------------------------------- Jobs ---------------------------------------
all_sync_job = define_asset_job(name = "all_sync_job", selection = "*" )
#------------------------------- Schedules ----------------------------------

#------------------------------- Definitions --------------------------------
defs = Definitions(
    assets=all_assets,
    jobs=[all_sync_job],
    resources={
        "mlflow": mlflow_tracking.configured({
            "mlflow_tracking_uri": "http://127.0.0.1:5000",
            "experiment_name": "boston_project"
        })
    }
)
