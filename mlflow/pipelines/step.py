import abc
import logging
import yaml
from typing import TypeVar, Dict, Any

import mlflow
from mlflow.pipelines.utils import get_pipeline_name, get_pipeline_config
from mlflow.utils.databricks_utils import is_in_databricks_runtime

_logger = logging.getLogger(__name__)

StepType = TypeVar("StepType", bound="BaseStep")


class BaseStep(metaclass=abc.ABCMeta):

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
        self.pipeline_config = get_pipeline_config(pipeline_root_path=pipeline_root)

    def run(self, output_directory: str):
        """
        Executes the step by running common setup operations and invoking
        step-specific code (as defined in ``_run()``).

        :param output_directory: String file path to the directory where step
                                 outputs should be stored.
        :return: Results from executing the corresponding step.
        """
        self._initialize_databricks_pyspark_connection_if_applicable()
        self._run(output_directory)
        return self.inspect(output_directory)

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
    def inspect(self, output_directory: str):
        """
        Inspect the step output state that was stored as part of the last execution.
        Each individual step needs to implement this function to return a materialized
        output to display to the end user.

        :param output_directory: String file path where to the directory where step
                                 outputs are located.
        :return: Results from the last execution of the corresponding step.
        """
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
    def environment(self):
        return {}

    def _initialize_databricks_pyspark_connection_if_applicable(self) -> None:
        """
        Initializes a connection to the Databricks PySpark Gateway if MLflow Pipelines is running
        in the Databricks Runtime.
        """
        if is_in_databricks_runtime():
            try:
                from dbruntime.spark_connection import initialize_spark_connection, is_pinn_mode_enabled
                initialize_spark_connection(is_pinn_mode_enabled())
            except Exception as e:
                _logger.warning(
                    "Encountered unexpected failure while initializing Spark connection. Spark"
                    " operations may not succeed. Exception: %s", e
                )

