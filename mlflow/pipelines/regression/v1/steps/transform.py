import importlib
import logging
import os
import sys

import cloudpickle

from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.utils.file_utils import read_yaml

_logger = logging.getLogger(__name__)


class TransformStep(BaseStep):
    def __init__(self, step_config, pipeline_root):
        super().__init__(step_config, pipeline_root)
        self.target_column = self.pipeline_config.get("target_col")
        (self.transformer_module_name, self.transformer_method_name,) = self.step_config[
            "transformer_method"
        ].rsplit(".", 1)

    def _run(self, output_directory):
        import pandas as pd

        train_data_path = get_step_output_path(
            pipeline_name=self.pipeline_name,
            step_name="split",
            relative_path="train.parquet",
        )
        train_df = pd.read_parquet(train_data_path)

        validation_data_path = get_step_output_path(
            pipeline_name=self.pipeline_name,
            step_name="split",
            relative_path="validation.parquet",
        )
        validation_df = pd.read_parquet(validation_data_path)

        sys.path.append(self.pipeline_root)
        transformer_fn = getattr(
            importlib.import_module(self.transformer_module_name), self.transformer_method_name
        )
        transformer = transformer_fn()
        transformer.fit(train_df)

        def process_dataset(dataset):
            features = dataset.drop(columns=[self.target_column])
            labels = dataset[self.target_column]
            transformed_feature_array = transformer.transform(features)
            num_features = transformed_feature_array.shape[1]
            df = pd.DataFrame(transformed_feature_array, columns=[f"feature_{i}" for i in range(num_features)])
            df[self.target_column] = labels.values
            return df

        train_transformed = process_dataset(train_df)
        validation_transformed = process_dataset(validation_df)
        """
        desired features are implied in the way the transformer is written. We give the train step a transformed
        dataset, so it also only sees desired columns. Thus desired columns don't need to be explicitly specified
        in pipeline.yaml.
        This means we would need to output the schema from dataset BEFORE transformation
        """

        with open(os.path.join(output_directory, "transformer.pkl"), "wb") as f:
            cloudpickle.dump(transformer, f)

        train_transformed.to_parquet(os.path.join(output_directory, "train_transformed.parquet"))
        validation_transformed.to_parquet(os.path.join(output_directory, "validation_transformed.parquet"))

    def _inspect(self, output_directory):
        # Do step-specific code to inspect/materialize the output of the step
        _logger.info("transform inspect code %s", output_directory)
        pass

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        step_config = read_yaml(os.path.join(pipeline_root, "steps"), "transformer_config.yaml")
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "transform"
