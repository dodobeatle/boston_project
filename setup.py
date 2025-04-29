from setuptools import find_packages, setup

setup(
    name="boston_project",
    packages=find_packages(exclude=["boston_project_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
