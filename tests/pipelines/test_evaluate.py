import sys
from unittest import mock
from pathlib import Path
import shutil

import pytest
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

import mlflow
from mlflow.utils.file_utils import read_yaml
from mlflow.pipelines.utils import _PIPELINE_CONFIG_FILE_NAME
from mlflow.pipelines.utils.execution import _MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR
from mlflow.pipelines.regression.v1.steps.split import _OUTPUT_TEST_FILE_NAME
from mlflow.pipelines.regression.v1.steps.evaluate import EvaluateStep
from mlflow.exceptions import MlflowException


@pytest.fixture(autouse=True)
def tmp_pipeline_exec_path(monkeypatch, tmp_path) -> Path:
    path = tmp_path.joinpath("pipeline_execution")
    path.mkdir(parents=True)
    monkeypatch.setenv(_MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR, str(path))
    yield path
    shutil.rmtree(path)


@pytest.fixture
def tmp_pipeline_root_path(tmp_path) -> Path:
    path = tmp_path.joinpath("pipeline_root")
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path)


@pytest.fixture(autouse=True)
def clear_custom_metrics_module_cache():
    key = "steps.custom_metrics"
    if key in sys.modules:
        del sys.modules[key]


def train_and_log_model():
    mlflow.set_experiment("demo")
    with mlflow.start_run() as run:
        X, y = load_diabetes(as_frame=True, return_X_y=True)
        model = LinearRegression().fit(X, y)
        mlflow.sklearn.log_model(model, artifact_path="model")
    return run.info.run_id


@pytest.mark.parametrize("mae_threshold", [-1, 1_000_000])
def test_evaluate_step_run(
    tmp_pipeline_root_path: Path, tmp_pipeline_exec_path: Path, mae_threshold: int
):
    split_step_output_dir = tmp_pipeline_exec_path.joinpath("steps", "split", "outputs")
    split_step_output_dir.mkdir(parents=True)
    X, y = load_diabetes(as_frame=True, return_X_y=True)
    test_df = X.assign(y=y).sample(n=100, random_state=42)
    test_df.to_parquet(split_step_output_dir.joinpath(_OUTPUT_TEST_FILE_NAME))

    run_id = train_and_log_model()
    train_step_output_dir = tmp_pipeline_exec_path.joinpath("steps", "train", "outputs")
    train_step_output_dir.mkdir(parents=True)
    train_step_output_dir.joinpath("run_id").write_text(run_id)

    evaluate_step_output_dir = tmp_pipeline_exec_path.joinpath("steps", "evaluate", "outputs")
    evaluate_step_output_dir.mkdir(parents=True)

    pipeline_yaml = tmp_pipeline_root_path.joinpath(_PIPELINE_CONFIG_FILE_NAME)
    pipeline_yaml.write_text(
        """
template: "regression/v1"
target_col: "y"
steps:
  evaluate:
    validation_criteria:
      - metric: root_mean_squared_error
        threshold: 1_000_000
      - metric: mean_absolute_error
        threshold: {mae_threshold}
      - metric: weighted_mean_squared_error
        threshold: 1_000_000
metrics:
  custom:
    - name: weighted_mean_squared_error
      function: weighted_mean_squared_error
      greater_is_better: False
""".format(
            mae_threshold=mae_threshold
        )
    )
    pipeline_steps_dir = tmp_pipeline_root_path.joinpath("steps")
    pipeline_steps_dir.mkdir(parents=True)
    pipeline_steps_dir.joinpath("custom_metrics.py").write_text(
        """
def weighted_mean_squared_error(eval_df, builtin_metrics):
    from sklearn.metrics import mean_squared_error

    return {
        "weighted_mean_squared_error": mean_squared_error(
            eval_df["prediction"],
            eval_df["target"],
            sample_weight=1 / eval_df["prediction"].values,
        )
    }
"""
    )
    pipeline_config = read_yaml(tmp_pipeline_root_path, _PIPELINE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_pipeline_config(pipeline_config, str(tmp_pipeline_root_path))
    evaluate_step._run(str(evaluate_step_output_dir))

    logged_metrics = mlflow.tracking.MlflowClient().get_run(run_id).data.metrics
    logged_metrics = {k.replace("_on_data_test", ""): v for k, v in logged_metrics.items()}
    assert "weighted_mean_squared_error" in logged_metrics
    model_validation_status_path = evaluate_step_output_dir.joinpath("model_validation_status")
    assert model_validation_status_path.exists()
    expected_status = "REJECTED" if mae_threshold < 0 else "VALIDATED"
    assert model_validation_status_path.read_text() == expected_status


def test_validation_criteria_contain_undefined_metrics(tmp_pipeline_root_path: Path):
    pipeline_yaml = tmp_pipeline_root_path.joinpath(_PIPELINE_CONFIG_FILE_NAME)
    pipeline_yaml.write_text(
        """
template: "regression/v1"
target_col: "y"
steps:
  evaluate:
    validation_criteria:
      - metric: root_mean_squared_error
        threshold: 100
      - metric: undefined_metric
        threshold: 100
"""
    )
    pipeline_steps_dir = tmp_pipeline_root_path.joinpath("steps")
    pipeline_steps_dir.mkdir(parents=True)

    pipeline_config = read_yaml(tmp_pipeline_root_path, _PIPELINE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_pipeline_config(pipeline_config, str(tmp_pipeline_root_path))
    with pytest.raises(
        MlflowException,
        match=r"Validation criteria contain undefined metrics: \['undefined_metric'\]",
    ):
        evaluate_step._validate_validation_criteria()


def test_custom_metric_function_does_not_exist(tmp_pipeline_root_path: Path):
    pipeline_yaml = tmp_pipeline_root_path.joinpath(_PIPELINE_CONFIG_FILE_NAME)
    pipeline_yaml.write_text(
        """
template: "regression/v1"
target_col: "y"
steps:
  evaluate:
    validation_criteria:
      - metric: weighted_mean_squared_error
        threshold: 100
metrics:
  custom:
    - name: weighted_mean_squared_error
      function: weighted_mean_squared_error
      greater_is_better: False
"""
    )
    pipeline_steps_dir = tmp_pipeline_root_path.joinpath("steps")
    pipeline_steps_dir.mkdir(parents=True)
    pipeline_steps_dir.joinpath("custom_metrics.py").write_text(
        """
def one(eval_df, builtin_metrics):
    return {"one": 1}
"""
    )
    pipeline_config = read_yaml(tmp_pipeline_root_path, _PIPELINE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_pipeline_config(pipeline_config, str(tmp_pipeline_root_path))
    with pytest.raises(MlflowException, match="Failed to load custom metric functions") as exc:
        evaluate_step._load_custom_metric_functions()
    assert isinstance(exc.value.__cause__, AttributeError)
    assert "weighted_mean_squared_error" in str(exc.value.__cause__)


def test_custom_metrics_module_does_not_exist(tmp_pipeline_root_path: Path):
    pipeline_yaml = tmp_pipeline_root_path.joinpath(_PIPELINE_CONFIG_FILE_NAME)
    pipeline_yaml.write_text(
        """
template: "regression/v1"
target_col: "y"
steps:
  evaluate:
    validation_criteria:
      - metric: weighted_mean_squared_error
        threshold: 100
metrics:
  custom:
    - name: weighted_mean_squared_error
      function: weighted_mean_squared_error
      greater_is_better: False
"""
    )
    pipeline_steps_dir = tmp_pipeline_root_path.joinpath("steps")
    pipeline_steps_dir.mkdir(parents=True)

    pipeline_config = read_yaml(tmp_pipeline_root_path, _PIPELINE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_pipeline_config(pipeline_config, str(tmp_pipeline_root_path))
    with pytest.raises(MlflowException, match="Failed to load custom metric functions") as exc:
        evaluate_step._load_custom_metric_functions()
    assert isinstance(exc.value.__cause__, ModuleNotFoundError)
    assert "No module named 'steps.custom_metrics'" in str(exc.value.__cause__)


def test_custom_metrics_override_builtin_metrics(
    tmp_pipeline_root_path: Path, tmp_pipeline_exec_path: Path
):
    split_step_output_dir = tmp_pipeline_exec_path.joinpath("steps", "split", "outputs")
    split_step_output_dir.mkdir(parents=True)
    X, y = load_diabetes(as_frame=True, return_X_y=True)
    test_df = X.assign(y=y).sample(n=100, random_state=42)
    test_df.to_parquet(split_step_output_dir.joinpath(_OUTPUT_TEST_FILE_NAME))

    run_id = train_and_log_model()
    train_step_output_dir = tmp_pipeline_exec_path.joinpath("steps", "train", "outputs")
    train_step_output_dir.mkdir(parents=True)
    train_step_output_dir.joinpath("run_id").write_text(run_id)

    evaluate_step_output_dir = tmp_pipeline_exec_path.joinpath("steps", "evaluate", "outputs")
    evaluate_step_output_dir.mkdir(parents=True)

    pipeline_yaml = tmp_pipeline_root_path.joinpath(_PIPELINE_CONFIG_FILE_NAME)
    pipeline_yaml.write_text(
        """
template: "regression/v1"
target_col: "y"
steps:
  evaluate:
    validation_criteria:
      - metric: root_mean_squared_error
        threshold: 10
      - metric: mean_absolute_error
        threshold: 10
metrics:
  custom:
    - name: mean_absolute_error
      function: mean_absolute_error
      greater_is_better: False
    - name: root_mean_squared_error
      function: root_mean_squared_error
      greater_is_better: False
"""
    )
    pipeline_steps_dir = tmp_pipeline_root_path.joinpath("steps")
    pipeline_steps_dir.mkdir(parents=True)
    pipeline_steps_dir.joinpath("custom_metrics.py").write_text(
        """
def mean_absolute_error(eval_df, builtin_metrics):
    return {"mean_absolute_error": 1}

def root_mean_squared_error(eval_df, builtin_metrics):
    return {"root_mean_squared_error": 1}
"""
    )
    pipeline_config = read_yaml(tmp_pipeline_root_path, _PIPELINE_CONFIG_FILE_NAME)
    evaluate_step = EvaluateStep.from_pipeline_config(pipeline_config, str(tmp_pipeline_root_path))

    with mock.patch(
        "mlflow.pipelines.regression.v1.steps.evaluate._logger.warning"
    ) as mock_warning:
        evaluate_step._run(str(evaluate_step_output_dir))
        mock_warning.assert_called_once_with(
            "Custom metrics overrode the following built-in metrics: %s",
            ["mean_absolute_error", "root_mean_squared_error"],
        )
    logged_metrics = mlflow.tracking.MlflowClient().get_run(run_id).data.metrics
    logged_metrics = {k.replace("_on_data_test", ""): v for k, v in logged_metrics.items()}
    assert "root_mean_squared_error" in logged_metrics
    assert logged_metrics["root_mean_squared_error"] == 1
    assert "mean_absolute_error" in logged_metrics
    assert logged_metrics["mean_absolute_error"] == 1
    model_validation_status_path = evaluate_step_output_dir.joinpath("model_validation_status")
    assert model_validation_status_path.exists()
    assert model_validation_status_path.read_text() == "VALIDATED"
