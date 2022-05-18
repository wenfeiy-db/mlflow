import os
from unittest import mock

import pandas as pd
import pytest
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

import mlflow
from mlflow.utils.file_utils import read_yaml
from mlflow.pipelines.utils import _PIPELINE_CONFIG_FILE_NAME
from mlflow.pipelines.utils.execution import _MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR
from mlflow.pipelines.regression.v1.steps.split import _OUTPUT_TEST_FILE_NAME
from mlflow.pipelines.regression.v1.steps.evaluate import EvaluateStep


@pytest.fixture
def temp_pipelines_execution_directory(monkeypatch, tmp_path):
    monkeypatch.setenv(_MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR, str(tmp_path))


use_tmp_pipelines_execution_directory = pytest.mark.usefixtures(
    temp_pipelines_execution_directory.__name__
)


def train_and_log_model():
    mlflow.set_experiment("demo")
    with mlflow.start_run() as run:
        X, y = load_diabetes(as_frame=True, return_X_y=True)
        model = LinearRegression().fit(X, y)
        mlflow.sklearn.log_model(model, artifact_path="model")
    return run.info.run_id


@use_tmp_pipelines_execution_directory
def test_evaluate_step_run(tmp_path):
    split_step_output_dir = tmp_path.joinpath("steps", "split", "outputs")
    split_step_output_dir.mkdir(parents=True)
    X, y = load_diabetes(as_frame=True, return_X_y=True)
    test_df = X.assign(y=y).sample(n=100, random_state=42)
    test_df.to_parquet(split_step_output_dir.joinpath(_OUTPUT_TEST_FILE_NAME))

    run_id = train_and_log_model()
    train_step_output_dir = tmp_path.joinpath("steps", "train", "outputs")
    train_step_output_dir.mkdir(parents=True)
    train_step_output_dir.joinpath("run_id").write_text(run_id)

    evaluate_step_output_dir = tmp_path.joinpath("steps", "evaluate", "outputs")
    evaluate_step_output_dir.mkdir(parents=True)

    pipeline_root = tmp_path.joinpath("pipeline_root")
    pipeline_root.mkdir(parents=True)
    pipeline_yaml = pipeline_root.joinpath(_PIPELINE_CONFIG_FILE_NAME)
    pipeline_yaml.write_text(
        """
template: "regression/v1"
target_col: "y"
steps:
  evaluate:
    validation_criteria:
      - metric: root_mean_squared_error
        threshold: 0.75
      - metric: mean_absolute_error
        threshold: 50
      - metric: weighted_mean_squared_error
        threshold: 0.5
"""
    )
    pipeline_config = read_yaml(pipeline_root, _PIPELINE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_pipeline_config(pipeline_config, str(pipeline_root))
    evaluate_step._run(str(evaluate_step_output_dir))
