import abc
import json
import logging
import os
import shutil
import subprocess
import time
import traceback
import yaml
from enum import Enum
from typing import TypeVar, Dict, Any

from mlflow.pipelines.cards import BaseCard, CARD_PICKLE_NAME, FailureCard, CARD_HTML_NAME
from mlflow.pipelines.utils import get_pipeline_name
from mlflow.tracking import MlflowClient
from mlflow.utils.databricks_utils import (
    is_in_databricks_runtime,
    is_running_in_ipython_environment,
)


_logger = logging.getLogger(__name__)


class StepStatus(Enum):
    UNKNOWN = "UNKNOWN"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class StepExecutionState:
    _KEY_STATUS = "pipeline_step_execution_status"
    _KEY_LAST_UPDATED_TIMESTAMP = "pipeline_step_execution_last_updated_timestamp"

    def __init__(self, status: StepStatus, last_updated_timestamp: int):
        self.status = status
        self.last_updated_timestamp = last_updated_timestamp

    def to_dict(self):
        return {
            StepExecutionState._KEY_STATUS: self.status.value,
            StepExecutionState._KEY_LAST_UPDATED_TIMESTAMP: self.last_updated_timestamp,
        }

    @classmethod
    def from_dict(cls, status_dict):
        return cls(
            status=StepStatus[status_dict[StepExecutionState._KEY_STATUS]],
            last_updated_timestamp=status_dict[StepExecutionState._KEY_LAST_UPDATED_TIMESTAMP],
        )


StepType = TypeVar("StepType", bound="BaseStep")


class BaseStep(metaclass=abc.ABCMeta):
    _EXECUTION_STATE_FILE_NAME = "execution_state.json"

    def __init__(self, step_config: Dict[str, Any], pipeline_root: str):
        """
        :param step_config: dictionary of the config needed to
                            run/implement the step.
        :param pipeline_root: String file path to the directory where step
                              are defined.
        """
        self.step_config = step_config
        self.pipeline_root = pipeline_root
        self.pipeline_name = get_pipeline_name(pipeline_root_path=pipeline_root)

    def run(self, output_directory: str):
        """
        Executes the step by running common setup operations and invoking
        step-specific code (as defined in ``_run()``).

        :param output_directory: String file path to the directory where step
                                 outputs should be stored.
        :return: Results from executing the corresponding step.
        """
        self._initialize_databricks_spark_connection_and_hooks_if_applicable()
        try:
            self._update_status(status=StepStatus.RUNNING, output_directory=output_directory)
            step_card = self._run(output_directory=output_directory)
            self._update_status(status=StepStatus.SUCCEEDED, output_directory=output_directory)
        except Exception:
            self._update_status(status=StepStatus.FAILED, output_directory=output_directory)
            step_card = FailureCard(
                pipeline_name=self.pipeline_name,
                step_name=self.name,
                failure_traceback=traceback.format_exc(),
            )
            raise
        finally:
            step_card.save(path=output_directory)
            step_card.save_as_html(path=output_directory)

    def inspect(self, output_directory: str):
        """
        Inspect the step output state by running the generic inspect information here and
        running the step specific inspection code in the step's _inspect() method.

        :param output_directory: String file path where to the directory where step
                                 outputs are located.
        :return: None
        """
        card_path = os.path.join(output_directory, CARD_PICKLE_NAME)
        if not os.path.exists(card_path):
            return None

        card = BaseCard.load(card_path)
        if is_running_in_ipython_environment():
            card.display()
        else:
            card_html_path = os.path.join(output_directory, CARD_HTML_NAME)
            if os.path.exists(card_html_path) and shutil.which("open") is not None:
                subprocess.run(["open", card_html_path], check=True)

    @abc.abstractmethod
    def _run(self, output_directory: str):
        """
        This function is responsible for executing the step, writing outputs
        to the specified directory, and returning results to the user. It
        is invoked by the internal step runner.

        :param output_directory: String file path to the directory where step outputs
                                 should be stored.
        :return: Results from executing the corresponding step.
        """
        pass

    def clean(self) -> None:
        pass

    @classmethod
    @abc.abstractmethod
    def from_pipeline_config(cls, pipeline_config: Dict[str, Any], pipeline_root: str) -> StepType:
        """
        Constructs a step class instance by creating a step config using the pipeline
        config.
        Subclasses must implement this method to produce the config required to correctly
        run the corresponding step.

        :param pipeline_config: Dictionary representation of the full pipeline config.
        :param pipeline_root: String file path to the pipeline root directory.
        :return: class instance of the step.
        """
        pass

    @classmethod
    def from_step_config_path(cls, step_config_path: str, pipeline_root: str) -> StepType:
        """
        Constructs a step class instance using the config specified in the
        configuration file.

        :param step_config_path: String path to the step-specific configuration
                                 on the local filesystem.
        :param pipeline_root: String path to the pipeline root directory on
                              the local filesystem.
        :return: class instance of the step.
        """
        with open(step_config_path, "r") as f:
            step_config = yaml.safe_load(f)
        return cls(step_config, pipeline_root)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Returns back the name of the step for the current class instance. This is used
        downstream by the execution engine to create step-specific directory structures.
        """
        pass

    @property
    def environment(self) -> Dict[str, str]:
        """
        Returns environment variables associated with step that should be set when the
        step is executed.
        """
        return {}

    def get_execution_state(self, output_directory: str) -> StepExecutionState:
        execution_state_file_path = os.path.join(
            output_directory, BaseStep._EXECUTION_STATE_FILE_NAME
        )
        if os.path.exists(execution_state_file_path):
            with open(execution_state_file_path, "r") as f:
                return StepExecutionState.from_dict(json.load(f))
        else:
            return StepExecutionState(StepStatus.UNKNOWN, 0)

    def _update_status(self, status: StepStatus, output_directory: str) -> None:
        execution_state = StepExecutionState(status=status, last_updated_timestamp=time.time())
        with open(os.path.join(output_directory, BaseStep._EXECUTION_STATE_FILE_NAME), "w") as f:
            json.dump(execution_state.to_dict(), f)

    def _initialize_databricks_spark_connection_and_hooks_if_applicable(self) -> None:
        """
        Initializes a connection to the Databricks Spark Gateway and sets up associated hooks
        (e.g. MLflow Run creation notification hooks) if MLflow Pipelines is running in the
        Databricks Runtime.
        """
        if is_in_databricks_runtime():
            try:
                from dbruntime.spark_connection import (
                    initialize_spark_connection,
                    is_pinn_mode_enabled,
                )

                spark_handles, entry_point = initialize_spark_connection(is_pinn_mode_enabled())
            except Exception as e:
                _logger.warning(
                    "Encountered unexpected failure while initializing Spark connection. Spark"
                    " operations may not succeed. Exception: %s",
                    e,
                )
            else:
                try:
                    from dbruntime.MlflowCreateRunHook import get_mlflow_create_run_hook

                    # `get_mlflow_create_run_hook` sets up a patch to trigger a Databricks command
                    # notification every time an MLflow Run is created. This notification is
                    # visible to users in notebook environments
                    get_mlflow_create_run_hook(spark_handles["sc"], entry_point)
                except Exception as e:
                    _logger.warning(
                        "Encountered unexpected failure while setting up Databricks MLflow Run"
                        " creation hooks. Exception: %s",
                        e,
                    )

    def _log_step_card(self, run_id: str, step_name: str) -> None:
        """
        Logs a step card as an artifact (destination: <step_name>/card.html) in a specified run.
        If the step card does not exist, logging is skipped.

        :param run_id: Run ID to which the step card is logged.
        :param step_name: Step name.
        """
        from mlflow.pipelines.utils.execution import get_step_output_path

        local_card_path = get_step_output_path(
            pipeline_name=self.pipeline_name,
            step_name=step_name,
            relative_path=CARD_HTML_NAME,
        )
        if os.path.exists(local_card_path):
            MlflowClient().log_artifact(run_id, local_card_path, artifact_path=step_name)
