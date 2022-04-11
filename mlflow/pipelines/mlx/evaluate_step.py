import sys

import pandas as pd
from pandas_profiling import ProfileReport
import click

import cloudpickle
import shap
import matplotlib.pyplot as plt
from matplotlib import rcParams

import mlflow

@click.command()
@click.option('--pipeline-path', help='Path to pipeline pickle file')
@click.option('--tracking-uri', help='MLflow tracking URI')
@click.option('--run-id', help='path to Run ID')
@click.option('--train-path', help="Path to training data")
@click.option('--evaluate-path', help='Output path of evaluation results')
def evaluate_step(pipeline_path, tracking_uri, run_id, train_path, evaluate_path):
    # TODO: read target column from config
    X_train = pd.read_parquet(train_path).drop(columns=['price'])
    with open(pipeline_path, 'rb') as f:
        pipeline = cloudpickle.load(f)

    mode = X_train.mode().iloc[0]

    background = shap.sample(X_train, 10, random_state=3).fillna(mode)
    sample = shap.sample(X_train, 10, random_state=12).fillna(mode)

    predict = lambda x: pipeline.predict(pd.DataFrame(x, columns=X_train.columns))
    evaluateer = shap.Kernelevaluateer(predict, background, link="identity")
    shap_values = evaluateer.shap_values(sample)

    # https://giters.com/slundberg/shap/issues/1916
    rcParams.update({'figure.autolayout': True})

    shap.summary_plot(shap_values, sample, show=False)
    plt.savefig(evaluate_path, format='svg')

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("demo")  # hardcoded

    with open(run_id, 'r') as f:
        run_id = f.read()
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(evaluate_path)

if __name__ == "__main__":
    evaluate_step()
