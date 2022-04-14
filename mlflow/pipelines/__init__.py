import sys
import shutil
import subprocess
import os

import pandas as pd
import numpy as np


def ingest():
    """
    Ingest data
    """
    _enter_repository_root()
    _run_ingest(reingest=True)

def split():
    """
    Split data
    """
    _enter_repository_root()
    _run_ingest(reingest=False)
    _run_make("split")

    print("== Showing summary of input data ==\n")
    _maybe_open("split_summary.html")

    print("Split data into train/test sets")

    print("== Summary of train data ==\n")
    print(pd.read_parquet("split_train.parquet").describe())

    print("== Summary of test data ==\n")
    print(pd.read_parquet("split_test.parquet").describe())


def transform():
    """
    Transform features
    """
    _enter_repository_root()
    _run_ingest(reingest=False)
    _run_make("transform")

    print("== Summary of transformed features ==\n")
    df = pd.read_parquet("transform_train_transformed.parquet")
    X = np.vstack(df["features"])
    print(pd.DataFrame(X).describe())


def train():
    """
    Train a model
    """
    _enter_repository_root()
    _run_ingest(reingest=False)
    _run_make("train")

    print("== Trained a model at train_pipeline.pkl ==\n")


def evaluate():
    """
    Evaluate a model (explanations included)
    """
    _enter_repository_root()
    _run_ingest(reingest=False)
    _run_make("evaluate")

    print("== Created the model card ==\n")
    _maybe_open("evaluate_explanations.html")

    print("== Produced evaluation metrics ==\n")
    _maybe_open("evaluate_metrics.json")


def clean():
    """
    Clean
    """
    _enter_repository_root()
    _run_make("clean")


def inspect():
    """
    Inspect specific steps or full pipeline DAG
    """
    raise NotImplementedError


def _run_in_subprocess_and_stream_results(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    for c in iter(lambda: process.stdout.read(1), b''):
        sys.stdout.buffer.write(c)

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(process.stderr.decode())


def _run_make(rule_name):
    _run_in_subprocess_and_stream_results(["make", rule_name])


def _maybe_open(path):
    assert os.path.exists(path), f"{path} does not exist"
    if shutil.which("open") is not None:
        subprocess.run(["open", path])
    else:
        print(f"Please open {path} manually.")


def _run_ingest(reingest=False):
    """
    :param reingest: If `True`, reingest data even if it has already been ingested previously.
                     If `False`, only ingest data even it has not previously been ingested.
    """
    pass


def _enter_repository_root():
    # Replace with gitpython later if necessary / possible, since this is
    # already an MLflow dependency
    # TODO: Figure out if this works on Databricks
    repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode("utf-8").rstrip("\n")
    os.chdir(repo_root)
