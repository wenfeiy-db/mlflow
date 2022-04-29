from mlflow.pipelines.step import BaseStep
import logging

_logger = logging.getLogger(__name__)


class SplitStep(BaseStep):
    def _run(self, output_directory):
        # Do step-specific code to execute the split step
        _logger.info("split run code %s", output_directory)

    def inspect(self, output_directory):
        # Do step-specific code to inspect/materialize the output of the step
        _logger.info("split inspect code %s", output_directory)
        pass

    def clean(self):
        # Do step-specific code to clean all the artifacts and paths output of the step
        _logger.info("split clean code")
        pass

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        step_config = {}
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "split"
