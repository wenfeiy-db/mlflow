import datetime
import logging
import os
import time
from typing import Dict, Any

import mlflow
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.exceptions import MlflowException, INVALID_PARAMETER_VALUE
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.tracking.client import MlflowClient

_logger = logging.getLogger(__name__)

_OUTPUT_CARD_FILE_NAME = "register-explanations.html"
_MODEL_REGISTRY_STATUS_RETRIES = 10


class RegisterStep(BaseStep):
    def __init__(self, step_config: Dict[str, Any], pipeline_root: str):
        super(RegisterStep, self).__init__(step_config, pipeline_root)
        self.status = "Unknown"
        self.run_end_time = None
        self.execution_duration = None
        self.num_dropped_rows = None
        self.final_status = None
        self.model_url = None
        self.model_uri = None
        self.model_details = None
        self.alerts = None

        if "name" not in self.step_config:
            raise MlflowException(
                "Missing 'name' config in register step config.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        self.register_model_name = self.step_config.get("name")
        self.allow_non_validated_model = self.step_config.get("allow_non_validated_model", False)

    def _run(self, output_directory):
        try:
            run_start_time = time.time()
            run_id_path = get_step_output_path(
                pipeline_name=self.pipeline_name,
                step_name="train",
                relative_path="run_id",
            )
            with open(run_id_path, "r") as f:
                run_id = f.read()

            model_validation_path = get_step_output_path(
                pipeline_name=self.pipeline_name,
                step_name="evaluate",
                relative_path="model_validation_status",
            )
            with open(model_validation_path, "r") as f:
                model_validation = f.read()

            artifact_path = "model"
            if model_validation is "VALIDATED" or (
                model_validation is "UNKNOWN" and self.allow_non_validated_model
            ):
                self.model_url = "https://figurethisout.com"
                self.model_uri = "runs:/{run_id}/{artifact_path}".format(
                    run_id=run_id, artifact_path=artifact_path
                )
                self.model_details = mlflow.register_model(
                    model_uri=self.model_uri, name=self.register_model_name
                )
                self.final_status = self._wait_until_not_pending(self.model_details.version)
                self.alerts = ""
            else:
                self.model_url = "-"
                self.model_uri = "-"
                self.final_status = "-"
                self.alerts = "Model registration skipped.  Please check the validation result from Evaluate step."

            self.status = "Done"
        except Exception:
            self.status = "Failed"
            raise
        finally:
            self.run_end_time = time.time()
            self.execution_duration = self.run_end_time - run_start_time
            try:
                self._build_card(output_directory)
            except Exception as e:
                # swallow exception raised during building profiles and card.
                _logger.warning(f"Build card failed: {repr(e)}")
                # When log level is DEBUG, also log the error stack trace.
                _logger.debug("", exc_info=True)

    def _wait_until_not_pending(self, model_version: str) -> ModelVersionStatus:
        client = MlflowClient()
        for _ in range(_MODEL_REGISTRY_STATUS_RETRIES):
            model_version_details = client.get_model_version(
                name=self.register_model_name,
                version=model_version,
            )
            status = ModelVersionStatus.from_string(model_version_details.status)
            if status != ModelVersionStatus.PENDING_REGISTRATION:
                return status
            time.sleep(1)
        return ModelVersionStatus.PENDING_REGISTRATION

    def _build_card(self, output_directory: str) -> None:
        from mlflow.pipelines.regression.v1.cards.register import RegisterCard

        # Build card
        card = RegisterCard()

        run_end_datetime = datetime.datetime.fromtimestamp(self.run_end_time)
        card.add_markdown(
            "RUN_END_TIMESTAMP",
            f"**Last run completed at:** `{run_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}`",
        )
        card.add_markdown(
            "EXECUTION_DURATION", f"**Execution duration (s):** `{self.execution_duration:.2f}`"
        )
        card.add_markdown("RUN_STATUS", f"**Run status:** `{self.status}`")
        card.add_markdown("MODEL_URI", f"**Model URI:** `{self.model_uri}`")
        card.add_markdown("ALERTS", f"**Alerts:** `{self.alerts}`")
        card.add_markdown("MODEL_URL", f"**Model URL:** `{self.model_url}`")
        with open(os.path.join(output_directory, _OUTPUT_CARD_FILE_NAME), "w") as f:
            f.write(card.to_html())

    def inspect(self, output_directory):
        # Do step-specific code to inspect/materialize the output of the step
        _logger.info("register inspect code %s", output_directory)
        pass

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        try:
            step_config = pipeline_config["steps"]["register"]
            step_config[
                RegisterStep._TRACKING_URI_CONFIG_KEY
            ] = "sqlite:///metadata/mlflow/mlruns.db"
        except KeyError:
            raise MlflowException(
                "Config for register step is not found.", error_code=INVALID_PARAMETER_VALUE
            )
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "register"
