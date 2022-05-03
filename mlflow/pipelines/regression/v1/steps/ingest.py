import logging
import os
import shutil

from mlflow.pipelines.step import BaseStep

_logger = logging.getLogger(__name__)


class IngestStep(BaseStep):

    def _run(self, output_directory, reingest=False):  # pylint: disable=arguments-differ
        """
        :param reingest: If `True`, reingest data even if it has already been ingested previously.
                         If `False`, only ingest data if it has not previously been ingested.
        """
        dataset_dst_path = os.path.join(output_directory, "dataset.parquet")
        if not os.path.exists(dataset_dst_path) or reingest:
            return

        dataset_src_path = os.path.join(self.pipeline_root, "datasets", "autos.parquet")
        shutil.copyfile(dataset_src_path, dataset_dst_path)
        _logger.info("Resolved input data and stored it in '%s'", dataset_dst_path)

    def inspect(self, output_directory):
        # Do step-specific code to inspect/materialize the output of the step
        _logger.info("ingest inspect code %s", output_directory)
        pass

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        step_config = {}
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "ingest"
