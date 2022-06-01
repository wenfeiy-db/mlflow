import abc
import logging
import os
<<<<<<< HEAD
=======
import shutil
import subprocess
>>>>>>> origin/master
import yaml
from enum import Enum
from typing import TypeVar, Dict, Any

from mlflow.pipelines.utils import get_pipeline_name
from mlflow.utils.file_utils import path_to_local_file_uri
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


StepType = TypeVar("StepType", bound="BaseStep")


class BaseStep(metaclass=abc.ABCMeta):
    _STATUS_FILE_NAME = "status.txt"

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
        self.OUTPUT_CARD_FILE_NAME = None

    def run(self, output_directory: str):
        """
        Executes the step by running common setup operations and invoking
        step-specific code (as defined in ``_run()``).

        :param output_directory: String file path to the directory where step
                                 outputs should be stored.
        :return: Results from executing the corresponding step.
        """
<<<<<<< HEAD
        self._initialize_databricks_pyspark_connection_if_applicable()
        try:
            self._update_status(status=StepStatus.RUNNING, output_directory=output_directory)
            self._run(output_directory=output_directory)
            self._update_status(status=StepStatus.SUCCEEDED, output_directory=output_directory)
        except Exception:
            self._update_status(status=StepStatus.FAILED, output_directory=output_directory)
            raise

        return self.inspect(output_directory=output_directory)
=======
        self._initialize_databricks_spark_connection_and_hooks_if_applicable()
        self._run(output_directory)
        return self.inspect(output_directory)
>>>>>>> origin/master

    def inspect(self, output_directory: str):
        """
        Inspect the step output state by running the generic inspect information here and
        running the step specific inspection code in the step's _inspect() method.

        :param output_directory: String file path where to the directory where step
                                 outputs are located.
        :return: Results from the last execution of the corresponding step.
        """
        # Open the step card here
        from IPython.display import display, HTML

        if self.OUTPUT_CARD_FILE_NAME is not None:
            relative_path = os.path.join(output_directory, self.OUTPUT_CARD_FILE_NAME)
            output_filename = path_to_local_file_uri(os.path.abspath(relative_path))
            if is_running_in_ipython_environment():
                display(HTML(filename=output_filename))
            else:
                if shutil.which("open") is not None:
                    subprocess.run(["open", output_filename], check=True)

        return self._inspect(output_directory)

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

    @abc.abstractmethod
    def _inspect(self, output_directory: str):
        """
        Inspect the step output state that was stored as part of the last execution.
        Each individual step needs to implement this function to return a materialized
        output to display to the end user.

        :param output_directory: String file path where to the directory where step
                                 outputs are located.
        :return: Results from the last execution of the corresponding step.
        """
        pass

    def clean(self) -> None:
        pass

    def get_status(self, output_directory: str) -> StepStatus:
        status_file_path = os.path.join(output_directory, BaseStep._STATUS_FILE_NAME)
        if os.path.exists(status_file_path):
            with open(status_file_path, "r") as f:
                return StepStatus[f.read()]
        else:
            return StepStatus.UNKNOWN

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

    def _update_status(self, status: StepStatus, output_directory: str) -> None:
        with open(os.path.join(output_directory, BaseStep._STATUS_FILE_NAME), "w") as f:
            f.write(status.value)

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
