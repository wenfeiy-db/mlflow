import logging
import importlib.util
import sys
import operator
from pathlib import Path
from typing import Dict, Any

import cloudpickle

import mlflow
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.exceptions import MlflowException, INVALID_PARAMETER_VALUE

_logger = logging.getLogger(__name__)


# ref: https://stackoverflow.com/a/41595552/6943581
def _import_source_file(fname, modname):
    # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    spec = importlib.util.spec_from_file_location(modname, fname)
    if spec is None:
        raise ImportError(f"Could not load spec for module '{modname}' at: {fname}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    try:
        spec.loader.exec_module(module)
    except FileNotFoundError as e:
        raise ImportError(f"{e.strerror}: {fname}") from e
    return module


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
        self.metrics = self.step_config
        self.target_col = self.pipeline_config.get("target_col")
        self.status = "UNKNOWN"

    def _check_validation_criteria(self, metrics, validation_criteria):
        custom_metrics_gib_map = {
            m["name"]: m["greater_is_better"]
            for m in (self.step_config.get("metrics") or {}).get("custom", [])
        }
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

        custom_metrics_path = Path(self.pipeline_root, "steps", "custom_metrics.py")
        if custom_metrics_path.exists():
            custom_metrics_module = _import_source_file(custom_metrics_path, "custom_metrics")
            custom_metrics = [
                getattr(custom_metrics_module, cm["function"])
                for cm in self.step_config["metrics"].get("custom")
            ]
        else:
            custom_metrics = None

        with mlflow.start_run(run_id=run_id):
            model_uri = mlflow.get_artifact_uri("model")
            eval_result = mlflow.evaluate(
                model_uri,
                test_data,
                targets=self.target_col,
                model_type="regressor",
                evaluators="default",
                dataset_name="validation",
                custom_metrics=custom_metrics,
            )
            eval_result.save(output_directory)

        # Apply metric success criteria and log `is_validated` result
        validation_criteria = self.step_config.get("validation_criteria")
        if validation_criteria:
            criteria_summary = self._check_validation_criteria(
                eval_result.metrics, validation_criteria
            )
            self.status = "VALIDATED" if all(criteria_summary.values()) else "REJECTED"

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
