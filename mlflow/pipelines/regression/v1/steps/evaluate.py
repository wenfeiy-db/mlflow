import logging
import importlib.util
import sys
import operator
from pathlib import Path
from typing import Dict, Any

import mlflow
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, BAD_REQUEST
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.exceptions import MlflowException, INVALID_PARAMETER_VALUE

_logger = logging.getLogger(__name__)


_BUILTIN_METRIC_TO_GREATER_IS_BETTER = {
    # metric_name: greater_is_better
    "mean_absolute_error": False,
    "mean_squared_error": False,
    "root_mean_squared_error": False,
    "max_error": False,
    "mean_absolute_percentage_error": False,
}


class EvaluateStep(BaseStep):
    def __init__(self, step_config: Dict[str, Any], pipeline_root: str) -> None:
        super().__init__(step_config, pipeline_root)
        self.target_col = self.pipeline_config.get("target_col")
        self.status = "UNKNOWN"

    def _get_custom_metrics(self):
        return (self.step_config.get("metrics") or {}).get("custom")

    def _get_custom_metrics_gib_map(self):
        custom_metrics = self._get_custom_metrics()
        if not custom_metrics:
            return None
        return {cm["name"]: cm["greater_is_better"] for cm in custom_metrics}

    def _load_custom_metric_functions(self):
        try:
            sys.path.append(self.pipeline_root)
            custom_metrics_mod = importlib.import_module("steps.custom_metrics")
            return [
                getattr(custom_metrics_mod, cm["function"]) for cm in self._get_custom_metrics()
            ]
        except Exception as e:
            raise MlflowException(
                message="Failed to load custom metric functions",
                error_code=BAD_REQUEST,
            ) from e

    def _check_validation_criteria(self, metrics, validation_criteria):
        custom_metrics_gib_map = self._get_custom_metrics_gib_map() or {}
        gib_map = {**_BUILTIN_METRIC_TO_GREATER_IS_BETTER, **custom_metrics_gib_map}
        summary = {}
        for val_criterion in validation_criteria:
            metric_name = val_criterion["metric"]
            metric_val = metrics.get(metric_name)
            if metric_val is None:
                summary[metric_name] = False
                continue
            comp_func = operator.ge if gib_map[metric_name] else operator.le
            threshold = val_criterion["threshold"]
            summary[metric_name] = comp_func(metric_val, threshold)
        return summary

    def _run(self, output_directory):
        import pandas as pd

        test_data_path = get_step_output_path(
            pipeline_name=self.pipeline_name,
            step_name="split",
            relative_path="test.parquet",
        )
        test_data = pd.read_parquet(test_data_path)

        run_id_path = get_step_output_path(
            pipeline_name=self.pipeline_name,
            step_name="train",
            relative_path="run_id",
        )
        with open(run_id_path, "r") as f:
            run_id = f.read()

        mlflow.set_experiment("demo")  # hardcoded

        with mlflow.start_run(run_id=run_id):
            model_uri = mlflow.get_artifact_uri("model")
            eval_result = mlflow.evaluate(
                model_uri,
                test_data,
                targets=self.target_col,
                model_type="regressor",
                evaluators="default",
                dataset_name="test",
                custom_metrics=self._load_custom_metric_functions(),
            )
            eval_result.save(output_directory)

        validation_criteria = self.step_config.get("validation_criteria")
        if validation_criteria:
            criteria_summary = self._check_validation_criteria(
                eval_result.metrics, validation_criteria
            )
            model_validation_status = "VALIDATED" if all(criteria_summary.values()) else "REJECTED"
        else:
            model_validation_status = "UNKNOWN"

        Path(output_directory, "model_validation_status").write_text(model_validation_status)
        self.status = "DONE"

    def inspect(self, output_directory):
        # Do step-specific code to inspect/materialize the output of the step
        _logger.info("evaluate inspect code %s", output_directory)
        pass

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        try:
            step_config = pipeline_config["steps"]["evaluate"]
        except KeyError:
            raise MlflowException(
                "Config for evaluate step is not found.", error_code=INVALID_PARAMETER_VALUE
            )
        step_config[EvaluateStep._TRACKING_URI_CONFIG_KEY] = "/tmp/mlruns"
        step_config["metrics"] = pipeline_config.get("metrics")
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "evaluate"
