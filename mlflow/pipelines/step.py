import abc
import os
import yaml
from typing import TypeVar, Dict, Any

import mlflow
from mlflow.pipelines.utils import get_pipeline_name, get_pipeline_config
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.databricks_utils import is_running_in_ipython_environment


StepType = TypeVar("StepType", bound="BaseStep")


class BaseStep(metaclass=abc.ABCMeta):
    _TRACKING_URI_CONFIG_KEY = "tracking_uri"

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
        self.OUTPUT_CARD_FILE_NAME = None

    def _set_tracking_uri(self) -> None:
        uri = self.step_config.get(self._TRACKING_URI_CONFIG_KEY)
        if uri is not None:
            mlflow.set_tracking_uri(uri)

    def run(self, output_directory: str):
        """
        Executes the step by running common setup operations and invoking
        step-specific code (as defined in ``_run()``).

        :param output_directory: String file path to the directory where step
                                 outputs should be stored.
        :return: Results from executing the corresponding step.
        """
        self._set_tracking_uri()
        # other common setup stuff for steps goes here
        self._run(output_directory)
        return self.inspect(output_directory)

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
                import webbrowser

                file_uri = output_filename
                webbrowser.open_new(file_uri)
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

    def clean(self) -> None:
        pass
