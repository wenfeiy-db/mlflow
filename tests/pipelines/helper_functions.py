import os
import pathlib
import shutil
from contextlib import contextmanager

import mlflow
from mlflow.pipelines.step import BaseStep
from mlflow.utils.file_utils import TempDir

import pytest

PIPELINE_EXAMPLE_PATH_ENV_VAR_FOR_TESTS = "_PIPELINE_EXAMPLE_PATH"
PIPELINE_EXAMPLE_PATH_FROM_MLFLOW_ROOT = "examples/pipelines/sklearn_regression"


@pytest.fixture
def enter_pipeline_example_directory():
    pipeline_example_path = os.environ.get(PIPELINE_EXAMPLE_PATH_ENV_VAR_FOR_TESTS)
    if pipeline_example_path is None:
        mlflow_repo_root_directory = pathlib.Path(mlflow.__file__).parent.parent
        pipeline_example_path = mlflow_repo_root_directory / PIPELINE_EXAMPLE_PATH_FROM_MLFLOW_ROOT

    with chdir(pipeline_example_path):
        yield pipeline_example_path


@pytest.fixture
def enter_test_pipeline_directory(enter_pipeline_example_directory):
    pipeline_example_root_path = enter_pipeline_example_directory

    with TempDir(chdr=True) as tmp:
        test_pipeline_path = tmp.path("test_pipeline")
        shutil.copytree(pipeline_example_root_path, test_pipeline_path)
        os.chdir(test_pipeline_path)
        yield os.getcwd()


@contextmanager
def chdir(directory_path):
    og_dir = os.getcwd()
    try:
        os.chdir(directory_path)
        yield
    finally:
        os.chdir(og_dir)


class BaseStepImplemented(BaseStep):
    def _run(self, output_directory):
        pass

    def inspect(self, output_directory):
        pass

    def clean(self):
        pass

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        pass

    @property
    def name(self):
        pass
