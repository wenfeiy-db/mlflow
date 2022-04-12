import sys
import json

import cloudpickle
import matplotlib.pyplot as plt
import pandas as pd
import shap
from matplotlib import rcParams
from pandas_profiling import ProfileReport
from sklearn.metrics import mean_squared_error

import mlflow


def _evaluate(pipeline, train_data, test_data, metrics_output_path, worst_train_examples_output_path):
    """
    :param pipeline: The [<transformer>, <trained_model>] pipeline
    :param train_data: The training dataset
    :param test_data: The test dataset
    :param metrics_output_path: Output path of evaluation metrics 
    :param worst_train_examples_output_path: Output path of worst-performing training examples 
    """
    train_rmse, train_worst = _evaluate_model_on_dataset(pipeline, train_data)
    test_rmse, _ = _evaluate_model_on_dataset(pipeline, test_data)

    metrics = [
        {"dataset": "train", "metric": "rmse", "value": train_rmse},
        {"dataset": "test", "metric": "rmse", "value": test_rmse},
    ]

    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f)
    
    train_worst.to_parquet(worst_train_examples_output_path)


def _evaluate_model_on_dataset(model, df: pd.DataFrame):
    # TODO: read from conf
    label_col = 'price'
    y_true = df[label_col]
    y_pred = model.predict(df.drop(columns=[label_col]))
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    df['_pred_'] = y_pred
    df['_error_'] = (y_true - y_pred).abs()
    worst = df.nlargest(20, columns=['_error_'])
    
    return rmse, worst


def _explain(pipeline, X_train, explanations_output_path):
    """
    :param pipeline: The [<transformer>, <trained_model>] pipeline
    :param X_train: Features from the training dataset
    :param explanations_output_path: Output path of model explanations
    """
    mode = X_train.mode().iloc[0]

    background = shap.sample(X_train, 10, random_state=3).fillna(mode)
    sample = shap.sample(X_train, 10, random_state=12).fillna(mode)

    predict = lambda x: pipeline.predict(pd.DataFrame(x, columns=X_train.columns))
    evaluateer = shap.KernelExplainer(predict, background, link="identity")
    shap_values = evaluateer.shap_values(sample)

    # https://giters.com/slundberg/shap/issues/1916
    rcParams.update({'figure.autolayout': True})

    shap.summary_plot(shap_values, sample, show=False)
    plt.savefig(explanations_output_path, format='svg')

    mlflow.log_artifact(explanations_output_path)

def run_evaluate_step(pipeline_path, tracking_uri, run_id_path, train_data_path, test_data_path, explanations_output_path, metrics_output_path, worst_train_examples_output_path):
    """
    :param pipeline_path: Path to the [<transformer>, <trained_model>] pipeline
    :param tracking_uri: The MLflow Trracking URI
    :param run_id_path: Path to file containing MLflow Run ID
    :param test_data_path: Path to training data
    :param test_data_path: Path to test data
    :param explanations_output_path: Output path of model explanations
    :param metrics_output_path: Output path of evaluation metrics 
    :param worst_train_examples_output_path: Output path of worst-performing training examples 
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("demo")  # hardcoded
    with open(pipeline_path, 'rb') as f:
        pipeline = cloudpickle.load(f)

    train_data = pd.read_parquet(train_data_path)
    test_data = pd.read_parquet(test_data_path)
    X_train = train_data_path.drop(columns=['price'])
    
    with open(run_id_path, 'r') as f:
        run_id = f.read()

    with mlflow.start_run(run_id=run_id):
        _explain(pipeline, X_train, explanations_output_path)
        _evaluate(pipeline, train_data, test_data, metrics_output_path, worst_train_examples_output_path)
