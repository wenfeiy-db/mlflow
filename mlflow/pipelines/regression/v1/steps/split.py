import logging
import os

from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.utils.execution import get_step_output_path


_logger = logging.getLogger(__name__)


class SplitStep(BaseStep):
    def _run(self, output_directory):
        import pandas as pd
        from pandas_profiling import ProfileReport

        ingested_data_path = get_step_output_path(
            pipeline_name=self.pipeline_name,
            step_name="ingest",
            relative_path="dataset.parquet",
        )
        df = pd.read_parquet(ingested_data_path)

        profile = ProfileReport(df, title="Summary of Input Dataset", minimal=True)
        profile.to_file(output_file=os.path.join(output_directory, "summary.html"))

        # Drop null values.
        # TODO: load from conf
        df = df.dropna(subset=["price"])

        hash_buckets = df.apply(lambda x: abs(hash(tuple(x))) % 100, axis=1)
        is_train = hash_buckets < 80
        train = df[is_train]
        test = df[~is_train]

        train.to_parquet(os.path.join(output_directory, "train.parquet"))
        test.to_parquet(os.path.join(output_directory, "test.parquet"))

    def inspect(self, output_directory):
        # Do step-specific code to inspect/materialize the output of the step
        _logger.info("split inspect code %s", output_directory)
        pass

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        step_config = {}
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "split"
