import sys
import cloudpickle
import click
import pandas as pd
import importlib
import os
import yaml

import numpy as np

import mlflow

from sklearn.pipeline import make_pipeline


def run_train_step(
    transformed_train_data_path,
    train_config_path,
    transformer_path,
    tracking_uri,
    pipeline_output_path,
    run_id_output_path,
):
    """
    :param transformed_train_data_path: Path to transformed training data
    :param train_config: Path to training configuration yaml
    :param transformer: Path to transformer from `transform` step
    :param tracking_uri: The MLflow Tracking URI
    :param pipeline_output_path: Output path of [<transformer>, <trained_model>] pipeline
    :param run_id_output_path: Output path of file containing MLflow Run ID
    """
    sys.path.append(os.curdir)
    module_name, method_name = (
        yaml.load(open(train_config_path, "r")).get("train_method").rsplit(".", 1)
    )
    train_fn = getattr(importlib.import_module(module_name), method_name)
    model = train_fn()

    df = pd.read_parquet(transformed_train_data_path)

    X = df["features"]
    y = df["target"]

    X = np.vstack(X)
    y = np.array(y)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("demo")  # hardcoded
    mlflow.autolog(log_models=False)

    with mlflow.start_run() as run:
        model.fit(X, y)

        with open(transformer_path, "rb") as f:
            transformer = cloudpickle.load(f)

        pipeline = make_pipeline(transformer, model)
        mlflow.sklearn.log_model(pipeline, "model")

        with open(run_id_output_path, "w") as f:
            f.write(run.info.run_id)

    with open(pipeline_output_path, "wb") as f:
        cloudpickle.dump(pipeline, f)
