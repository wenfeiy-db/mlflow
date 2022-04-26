import os
import pathlib

import mlflow

import pytest

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
        yield pipeline_example_path
    finally:
        os.chdir(og_dir)
