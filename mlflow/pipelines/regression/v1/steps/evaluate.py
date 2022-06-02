import logging
import importlib.util
import sys
import operator
import datetime
import os
import time
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from collections import namedtuple

import mlflow
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, BAD_REQUEST
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.pipelines.utils.tracking import (
    get_pipeline_tracking_config,
    apply_pipeline_tracking_config,
    TrackingConfig,
    get_run_tags_env_vars,
)
from mlflow.projects.utils import get_databricks_env_vars
from mlflow.exceptions import MlflowException

_logger = logging.getLogger(__name__)


_FEATURE_IMPORTANCE_PLOT_FILE = "feature_importance.png"


_BUILTIN_METRIC_TO_GREATER_IS_BETTER = {
    # metric_name: greater_is_better
    "mean_absolute_error": False,
    "mean_squared_error": False,
    "root_mean_squared_error": False,
    "max_error": False,
    "mean_absolute_percentage_error": False,
}

MetricValidationResult = namedtuple(
    "MetricValidationResult", ["metric", "greater_is_better", "value", "threshold", "validated"]
)


class EvaluateStep(BaseStep):
    def __init__(self, step_config: Dict[str, Any], pipeline_root: str) -> None:
        super().__init__(step_config, pipeline_root)
        self.tracking_config = TrackingConfig.from_dict(step_config)
        self.target_col = self.step_config.get("target_col")
        self.run_end_time = None
        self.execution_duration = None
        self.model_validation_status = "UNKNOWN"

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
        """
        return a list of `MetricValidationResult` tuple instances.
        """
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
        summary = []
        for val_criterion in validation_criteria:
            metric_name = val_criterion["metric"]
            metric_val = metrics.get(metric_name)
            if metric_val is None:
                summary[metric_name] = False
                continue
            greater_is_better = metric_to_greater_is_better[metric_name]
            comp_func = operator.ge if greater_is_better else operator.le
            threshold = val_criterion["threshold"]
            validated = comp_func(metric_val, threshold)
            summary.append(
                MetricValidationResult(
                    metric=metric_name,
                    greater_is_better=greater_is_better,
                    value=metric_val,
                    threshold=threshold,
                    validated=validated,
                )
            )
        return summary

    def _run(self, output_directory):
        run_start_time = time.time()
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

        apply_pipeline_tracking_config(self.tracking_config)

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
                evaluator_config={
                    "explainability_algorithm": "kernel",
                    "explainability_nsamples": 100,
                },
            )
            eval_result.save(output_directory)

        validation_criteria = self.step_config.get("validation_criteria")
        if validation_criteria:
            criteria_summary = self._check_validation_criteria(
                eval_result.metrics, validation_criteria
            )
            self.model_validation_status = (
                "VALIDATED" if all(cr.validated for cr in criteria_summary) else "REJECTED"
            )
        else:
            self.model_validation_status = "UNKNOWN"
            criteria_summary = None

        Path(output_directory, "model_validation_status").write_text(
            self.model_validation_status
        )

        self.run_end_time = time.time()
        self.execution_duration = self.run_end_time - run_start_time
        return self._build_profiles_and_card(
            model_uri, test_data, eval_result, criteria_summary, output_directory
        )

    def _build_profiles_and_card(
        self, model_uri, test_data, eval_result, criteria_summary, output_directory
    ):
        """
        Constructs data profiles of predictions and errors and a step card instance corresponding
        to the current evaluate step state.

        :param model_uri: the uri of the model being evaluated.
        :param test_data: the test split dataset used to evaluate the model.
        :param eval_result: the evaluation result on test dataset returned by `mlflow.evalaute`
        :param criteria_summary: a list of `MetricValidationResult` instances
        :param output_directory: output directory used by the evaluate step.
        """
        from mlflow.pipelines.regression.v1.cards.evaluate import EvaluateCard
        from pandas_profiling import ProfileReport

        # Build card
        card = EvaluateCard(self.pipeline_name, self.name)

        run_end_datetime = datetime.datetime.fromtimestamp(self.run_end_time)

        metric_lines = [f"* **{k}**: {v:.6g}" for k, v in eval_result.metrics.items()]
        card.add_markdown("METRICS", "\n".join(metric_lines))

        if criteria_summary is not None:
            criteria_summary_df = pd.DataFrame(criteria_summary)

            def criteria_table_row_format(row):
                color = "background-color: {}".format(
                    "lightgreen" if row.validated else "lightpink"
                )
                return (color,) * len(row)

            criteria_html = (
                criteria_summary_df.style.apply(criteria_table_row_format, axis=1)
                .hide_index()
                .format({"value": "{:.6g}", "threshold": "{:.6g}"})
                .to_html()
            )
            card.add_html("METRIC_VALIDATION_RESULTS", criteria_html)

        shap_bar_plot_path = os.path.join(
            output_directory, "artifacts", "shap_feature_importance_plot_on_data_test.png"
        )
        shap_bar_plot_res_path = card._add_resource_file(shap_bar_plot_path)
        shap_bar_plot_img = (
            f'<img src="{shap_bar_plot_res_path}" width="800" />'
            if os.path.exists(shap_bar_plot_path)
            else "Unavailable"
        )
        shap_beeswarm_plot_path = os.path.join(
            output_directory, "artifacts", "shap_beeswarm_plot_on_data_test.png"
        )
        shap_beeswarm_plot_res_path = card._add_resource_file(shap_beeswarm_plot_path)
        shap_beeswarm_plot_img = (
            f'<img src="{shap_beeswarm_plot_res_path}" width="800" />'
            if os.path.exists(shap_beeswarm_plot_path)
            else "Unavailable"
        )

        card._add_tab(
            "Feature importance with raw features",
            '<h3 class="section-title">Shap bar plot</h3>{SHAP_BAR_PLOT}'
            '<h3 class="section-title">Shap beeswarm plot</h3>{SHAP_BEESWARM_PLOT}',
            html_variables={
                "SHAP_BAR_PLOT": shap_bar_plot_img,
                "SHAP_BEESWARM_PLOT": shap_beeswarm_plot_img,
            },
        )

        # Constructs data profiles of predictions and errors
        model = mlflow.pyfunc.load_model(model_uri)
        target_data = test_data[self.target_col]
        prediction_result = model.predict(test_data.drop(self.target_col, axis=1))
        pred_and_error_df = pd.DataFrame(
            {
                "target": target_data,
                "prediction": prediction_result,
                "error": prediction_result - target_data,
            }
        )
        pred_and_error_df_profile = ProfileReport(
            pred_and_error_df, title="Profile of Prediction and Error Dataset", minimal=True
        )
        card.add_pandas_profile(
            "Profile of Prediction and Error Dataset", pred_and_error_df_profile
        )
        pred_and_error_df_profile.to_file(
            output_file=os.path.join(output_directory, "pred_and_error_profile.html")
        )

        execution_duration_text = f"**Execution duration (s):** `{self.execution_duration:.6g}`"
        card._add_tab(
            "Step run summary",
            "{EXECUTION_DURATION}<br>{RUN_END_TIMESTAMP}<br>{VALIDATION_STATUS}",
            markdown_variables={
                "EXECUTION_DURATION": execution_duration_text,
                "RUN_END_TIMESTAMP": f"**Last run completed at:** "
                f"`{run_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}`",
                "VALIDATION_STATUS": f"**Validation status:** `{self.model_validation_status}`",
            },
        )

        return card

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        try:
            step_config = pipeline_config["steps"].get("evaluate") or {}
        except KeyError:
            raise MlflowException(
                "Config for evaluate step is not found.", error_code=INVALID_PARAMETER_VALUE
            )
        step_config["metrics"] = pipeline_config.get("metrics")
        step_config["target_col"] = pipeline_config.get("target_col")
        step_config.update(
            get_pipeline_tracking_config(
                pipeline_root_path=pipeline_root,
                pipeline_config=pipeline_config,
            ).to_dict()
        )
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "evaluate"

    @property
    def environment(self):
        environ = get_databricks_env_vars(tracking_uri=self.tracking_config.tracking_uri)
        environ.update(get_run_tags_env_vars())
        return environ
