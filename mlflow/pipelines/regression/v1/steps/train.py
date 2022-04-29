from mlflow.pipelines.step import BaseStep
import logging

_logger = logging.getLogger(__name__)


class TrainStep(BaseStep):
    def _run(self, output_directory):
        # Do step-specific code to execute the train step
        _logger.info("train run code %s", output_directory)

    def inspect(self, output_directory):
        # Do step-specific code to inspect/materialize the output of the step
        _logger.info("train inspect code %s", output_directory)
        pass

    def clean(self):
        # Do step-specific code to clean all the artifacts and paths output of the step
        _logger.info("train clean code")
        pass

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        step_config = {}
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "train"
