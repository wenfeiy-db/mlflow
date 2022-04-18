import os
import pathlib
import subprocess

import pytest
from click.testing import CliRunner

import mlflow
import mlflow.pipelines
import mlflow.pipelines.cli as pipelines_cli

PIPELINE_EXAMPLE_PATH_ENV_VAR = "PIPELINE_EXAMPLE_PATH"
PIPELINE_EXAMPLE_PATH_FROM_MLFLOW_ROOT = "examples/pipelines/sklearn_regression"


@pytest.fixture
def enter_pipeline_example_directory():
    og_dir = os.getcwd()
    try:
        pipeline_example_path = os.environ.get(PIPELINE_EXAMPLE_PATH_ENV_VAR)
        if pipeline_example_path is None:
            mlflow_repo_root_directory = pathlib.Path(mlflow.__file__).parent.parent
            pipeline_example_path = (
                mlflow_repo_root_directory / PIPELINE_EXAMPLE_PATH_FROM_MLFLOW_ROOT
            )
        os.chdir(pipeline_example_path)
        yield
    finally:
        os.chdir(og_dir)


@pytest.fixture
def clean_up_pipeline():
    try:
        yield
    finally:
        mlflow.pipelines.clean()


@pytest.mark.usefixtures("enter_pipeline_example_directory", "clean_up_pipeline")
def test_pipelines_api_flow_completes_successfully():
    mlflow.pipelines.clean()
    mlflow.pipelines.evaluate()

    mlflow.pipelines.clean()
    mlflow.pipelines.ingest()
    mlflow.pipelines.split()
    mlflow.pipelines.transform()
    mlflow.pipelines.train()
    mlflow.pipelines.evaluate()


@pytest.mark.usefixtures("enter_pipeline_example_directory", "clean_up_pipeline")
def test_pipelines_cli_flow_completes_successfully():
    CliRunner().invoke(pipelines_cli.clean)
    CliRunner().invoke(pipelines_cli.evaluate)

    CliRunner().invoke(pipelines_cli.clean)
    CliRunner().invoke(pipelines_cli.ingest)
    CliRunner().invoke(pipelines_cli.split)
    CliRunner().invoke(pipelines_cli.transform)
    CliRunner().invoke(pipelines_cli.train)
    CliRunner().invoke(pipelines_cli.evaluate)
