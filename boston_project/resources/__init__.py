from dagster import EnvVar
from dagster_airbyte import AirbyteResource

airbyte_resource = AirbyteResource(
    host=EnvVar("AIRBYTE_HOST"),
    port=EnvVar("AIRBYTE_PORT"),
    username=EnvVar("AIRBYTE_USERNAME"),
    password=EnvVar("AIRBYTE_PASSWORD")
)

