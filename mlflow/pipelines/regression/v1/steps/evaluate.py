import logging
import importlib.util
import sys
import operator
from pathlib import Path
from typing import Dict, Any

import mlflow
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, BAD_REQUEST
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.utils import get_pipeline_tracking_config, TrackingConfig
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.projects.utils import get_databricks_env_vars
from mlflow.exceptions import MlflowException

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
    def __init__(self, step_config: Dict[str, Any], pipeline_root: str):
        super().__init__(step_config, pipeline_root)
        self.tracking_config = TrackingConfig.from_dict(step_config)
        self.target_col = self.pipeline_config.get("target_col")
        self.status = "UNKNOWN"

    def _get_custom_metrics(self):
        return (self.step_config.get("metrics") or {}).get("custom")

    def _get_custom_metric_greater_is_better(self):
        custom_metrics = self._get_custom_metrics()
        return (
            {cm["name"]: cm["greater_is_better"] for cm in custom_metrics} if custom_metrics else {}
        )

    def _load_custom_metric_functions(self):
        custom_metrics = self._get_custom_metrics()
        if not custom_metrics:
            return None
        try:
            sys.path.append(self.pipeline_root)
            custom_metrics_mod = importlib.import_module("steps.custom_metrics")
            return [getattr(custom_metrics_mod, cm["function"]) for cm in custom_metrics]
        except Exception as e:
            raise MlflowException(
                message="Failed to load custom metric functions",
                error_code=BAD_REQUEST,
            ) from e

    def _validate_validation_criteria(self):
        """
        Validates validation criteria don't contain undefined metrics
        """
        val_metrics = set(vc["metric"] for vc in self.step_config.get("validation_criteria", []))
        if not val_metrics:
            return
        builtin_metrics = set(_BUILTIN_METRIC_TO_GREATER_IS_BETTER.keys())
        custom_metrics = set(self._get_custom_metric_greater_is_better().keys())
        undefined_metrics = val_metrics.difference(builtin_metrics.union(custom_metrics))
        if undefined_metrics:
            raise MlflowException(
                f"Validation criteria contain undefined metrics: {sorted(undefined_metrics)}",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def _check_validation_criteria(self, metrics, validation_criteria):
        custom_metric_greater_is_better = self._get_custom_metric_greater_is_better()
        overridden_builtin_metrics = set(custom_metric_greater_is_better.keys()).intersection(
            _BUILTIN_METRIC_TO_GREATER_IS_BETTER.keys()
        )
        if overridden_builtin_metrics:
            _logger.warning(
                "Custom metrics overrode the following built-in metrics: %s",
                sorted(overridden_builtin_metrics),
            )
        metric_to_greater_is_better = {
            **_BUILTIN_METRIC_TO_GREATER_IS_BETTER,
            **custom_metric_greater_is_better,
        }
        summary = {}
        for val_criterion in validation_criteria:
            metric_name = val_criterion["metric"]
            metric_val = metrics.get(metric_name)
            if metric_val is None:
                summary[metric_name] = False
                continue
            comp_func = operator.ge if metric_to_greater_is_better[metric_name] else operator.le
            threshold = val_criterion["threshold"]
            summary[metric_name] = comp_func(metric_val, threshold)
        return summary

    def _run(self, output_directory):
        import pandas as pd

        self._validate_validation_criteria()

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

        mlflow.set_tracking_uri(self.tracking_config.tracking_uri)
        mlflow.set_experiment(
            experiment_name=self.tracking_config.experiment_name,
            experiment_id=self.tracking_config.experiment_id,
        )

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
            step_config = pipeline_config["steps"].get("evaluate") or {}
        except KeyError:
            raise MlflowException(
                "Config for evaluate step is not found.", error_code=INVALID_PARAMETER_VALUE
            )
        step_config[EvaluateStep._TRACKING_URI_CONFIG_KEY] = "/tmp/mlruns"
        step_config["metrics"] = pipeline_config.get("metrics")
        step_config.update(get_pipeline_tracking_config(pipeline_root_path=pipeline_root).to_dict())
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "evaluate"

    @property
    def environment(self):
        return get_databricks_env_vars(tracking_uri=self.tracking_config.tracking_uri)
