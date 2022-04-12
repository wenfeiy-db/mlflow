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

@click.command()
@click.option('--input-path', help='Path to input data')
@click.option("--train-config", help="Path to train config")
@click.option('--transformer-path', help='Output path of transformer')
@click.option('--tracking-uri', help='MLflow tracking URI')
@click.option('--pipeline-path', help='Output path to the fitted pipeline')
@click.option('--run-path', help='Output path for run')
def train_step(input_path, train_config, transformer_path, tracking_uri, pipeline_path, run_path):
    """
    Transform data using a transformer method.
    """
    sys.path.append(os.curdir)
    module_name, method_name = yaml.load(open(train_config, "r")).get("train_method").rsplit('.', 1)
    train_fn = getattr(importlib.import_module(module_name), method_name)
    model = train_fn()

    df = pd.read_parquet(input_path)

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
    
        with open(run_path, "w") as f:
            f.write(run.info.run_id)

    with open(pipeline_path, 'wb') as f:
        cloudpickle.dump(pipeline, f)

if __name__ == "__main__":
    train_step()
