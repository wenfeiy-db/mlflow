from mlflow.pipelines.step import BaseStep
import logging

_logger = logging.getLogger(__name__)


class IngestStep(BaseStep):
    def _run(self, output_directory):
        # Do step-specific code to execute the ingest step
        _logger.info("ingest run code %s", output_directory)

    def inspect(self, output_directory):
        # Do step-specific code to inspect/materialize the output of the step
        _logger.info("ingest inspect code %s", output_directory)
        pass

    def clean(self):
        # Do step-specific code to clean all the artifacts and paths output of the step
        _logger.info("ingest clean code")
        pass

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        step_config = {}
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "ingest"
