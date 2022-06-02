import importlib
import logging
import os
import sys

import cloudpickle

from mlflow.pipelines.regression.v1.cards.transform import TransformCard
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.utils.file_utils import read_yaml

_logger = logging.getLogger(__name__)


class TransformStep(BaseStep):
    def __init__(self, step_config, pipeline_root):
        super().__init__(step_config, pipeline_root)
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
        df = pd.read_parquet(train_data_path)

        sys.path.append(self.pipeline_root)
        transformer_fn = getattr(
            importlib.import_module(self.transformer_module_name), self.transformer_method_name
        )
        transformer = transformer_fn()

        # TODO: load from conf
        X = df.drop(columns=["price"])
        y = df["price"]

        features = transformer.fit_transform(X)

        with open(os.path.join(output_directory, "transformer.pkl"), "wb") as f:
            cloudpickle.dump(transformer, f)

        transformed = pd.DataFrame(data={"features": list(features), "target": y})
        transformed.to_parquet(os.path.join(output_directory, "train_transformed.parquet"))

        return TransformCard(self.pipeline_name, self.name)

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        step_config = read_yaml(os.path.join(pipeline_root, "steps"), "transformer_config.yaml")
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "transform"
