import importlib
import logging
import os
import sys

import cloudpickle

import mlflow
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.utils.file_utils import read_yaml

_logger = logging.getLogger(__name__)


class TrainStep(BaseStep):
    def __init__(self, step_config, pipeline_root):
        super().__init__(step_config, pipeline_root)
        self.train_module_name, self.train_method_name = self.step_config["train_method"].rsplit(
            ".", 1
        )

    def _run(self, output_directory):
        import pandas as pd
        import numpy as np
        from sklearn.pipeline import make_pipeline

        transformed_train_data_path = get_step_output_path(
            pipeline_name=self.pipeline_name,
            step_name="transform",
            relative_path="train_transformed.parquet",
        )
        df = pd.read_parquet(transformed_train_data_path)

        transformer_path = get_step_output_path(
            pipeline_name=self.pipeline_name,
            step_name="transform",
            relative_path="transformer.pkl",
        )

        sys.path.append(self.pipeline_root)
        train_fn = getattr(importlib.import_module(self.train_module_name), self.train_method_name)
        model = train_fn()

        X = df["features"]
        y = df["target"]

        X = np.vstack(X)
        y = np.array(y)

        mlflow.set_experiment("demo")  # hardcoded
        mlflow.autolog(log_models=False)

        with mlflow.start_run() as run:
            model.fit(X, y)

            with open(transformer_path, "rb") as f:
                transformer = cloudpickle.load(f)

            if hasattr(model, "best_score_"):
                mlflow.log_metric("best_cv_score", model.best_score_)

            if hasattr(model, "best_params_"):
                mlflow.log_params(model.best_params_)

            pipeline = make_pipeline(transformer, model)
            mlflow.sklearn.log_model(pipeline, "model")

            with open(os.path.join(output_directory, "run_id"), "w") as f:
                f.write(run.info.run_id)

        with open(os.path.join(output_directory, "pipeline.pkl"), "wb") as f:
            cloudpickle.dump(pipeline, f)

            # Do step-specific code to execute the train step
            _logger.info("train run code %s", output_directory)

    def inspect(self, output_directory):
        # Do step-specific code to inspect/materialize the output of the step
        _logger.info("train inspect code %s", output_directory)
        pass

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        step_config = read_yaml(os.path.join(pipeline_root, "steps"), "train_config.yaml")
        step_config[TrainStep._TRACKING_URI_CONFIG_KEY] = "/tmp/mlruns"
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "train"
