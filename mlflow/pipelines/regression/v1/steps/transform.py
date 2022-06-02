import importlib
import logging
import os
import sys

import cloudpickle

from mlflow.exceptions import MlflowException, INVALID_PARAMETER_VALUE
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.pipelines.utils.tracking import get_pipeline_tracking_config
from mlflow.utils.file_utils import read_yaml

_logger = logging.getLogger(__name__)


class TransformStep(BaseStep):
    def __init__(self, step_config, pipeline_root):
        super().__init__(step_config, pipeline_root)
        self.target_col = self.step_config.get("target_col")
        (self.transformer_module_name, self.transformer_method_name,) = self.step_config[
            "transform_method"
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

        def transform_dataset(dataset):
            features = dataset.drop(columns=[self.target_col])
            labels = dataset[self.target_col]
            transformed_feature_array = transformer.transform(features)
            num_features = transformed_feature_array.shape[1]
            # TODO: get the correct feature names from the transformer
            df = pd.DataFrame(
                transformed_feature_array, columns=[f"feature_{i}" for i in range(num_features)]
            )
            df[self.target_col] = labels.values
            return df

        train_transformed = transform_dataset(train_df)
        validation_transformed = transform_dataset(validation_df)
        """
        desired features are implied in the way the transformer is written. We give the train step a transformed
        dataset, so it also only sees desired columns. Thus desired columns don't need to be explicitly specified
        in pipeline.yaml.
        This means we would need to output the schema from dataset BEFORE transformation
        """

        with open(os.path.join(output_directory, "transformer.pkl"), "wb") as f:
            cloudpickle.dump(transformer, f)

        train_transformed.to_parquet(os.path.join(output_directory, "train_transformed.parquet"))
        validation_transformed.to_parquet(
            os.path.join(output_directory, "validation_transformed.parquet")
        )

    def _inspect(self, output_directory):
        # Do step-specific code to inspect/materialize the output of the step
        _logger.info("transform inspect code %s", output_directory)
        pass

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        try:
            step_config = pipeline_config["steps"]["transform"]
            step_config.update(
                get_pipeline_tracking_config(
                    pipeline_root_path=pipeline_root,
                    pipeline_config=pipeline_config,
                ).to_dict()
            )
        except KeyError:
            step_config = {}
        step_config["target_col"] = pipeline_config.get("target_col")
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "transform"
