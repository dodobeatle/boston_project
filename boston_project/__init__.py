from dagster import file_relative_path
import yaml

# Load the cofigs.yml file (config parameters file)
config_file_path = file_relative_path(__file__, "configs.yml")
with open(config_file_path, "r") as file:
    config_parameters = yaml.safe_load(file)