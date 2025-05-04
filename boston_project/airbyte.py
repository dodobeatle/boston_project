from dagster_airbyte import load_assets_from_airbyte_instance
from .resources import airbyte_resource

airbyte_connection = load_assets_from_airbyte_instance(airbyte_resource,
                                                    key_prefix="boston_project_raw", 
                                                    connection_to_group_fn=lambda connection_name: "raw_data_ingestion")
