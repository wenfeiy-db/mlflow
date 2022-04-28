from typing import TypeVar, Dict, Any
import mlflow
import abc
import mlflow.utils.file_utils


PipelineStep = TypeVar("PipelineStep", bound="BaseStep")


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
        return self._run(output_directory)

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

    @abc.abstractmethod
    def clean(self) -> None:
        """
        Remove the output of the step that was stored as part of the last
        execution. Each individual step needs to implement this function to
        clean its outputs.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def from_pipeline_config(
        cls, pipeline_config: Dict[str, Any], pipeline_root: str
    ) -> PipelineStep:
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
    def from_step_config_path(cls, step_config_path: str, pipeline_root: str) -> PipelineStep:
        """
        Constructs a step class instance using the config specified in the
        configuration file.

        :param step_config_path: String path to the step-specific configuration
                                 on the local filesystem.
        :param pipeline_root: String path to the pipeline root directory on
                              the local filesystem.
        :return: class instance of the step.
        """
        step_config = mlflow.utils.file_utils.read_yaml(pipeline_root, step_config_path)
        return cls(step_config, pipeline_root)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Returns back the name of the step for the current class instance. This is used
        downstream by the execution engine to create step-specific directory structures.
        """
        pass
