import os
import subprocess

import pytest
from click.testing import CliRunner

import mlflow.pipelines
from mlflow.pipelines import (
    ingest as ingest_cli,
    split as split_cli,
    transform as transform_cli,
    train as train_cli,
    evaluate as evaluate_cli,
)

PIPELINE_EXAMPLE_PATH_FROM_MLFLOW_ROOT = "examples/pipelines/example_pipeline"


@pytest.fixture
def enter_pipeline_example_directory():
    og_dir = os.getcwd()
    try:
        mlflow_repo_root_path = (
            subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
            .decode("utf-8")
            .rstrip("\n")
        )
        pipeline_example_path = os.path.join(
            mlflow_repo_root_path, PIPELINE_EXAMPLE_PATH_FROM_MLFLOW_ROOT
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
    CliRunner().invoke(ingest_cli)
