import sys
import shutil
import subprocess
import os

import click
import pandas as pd
import numpy as np


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

@click.group("pipelines")
def commands():
    pass

@commands.command(help='ingest data')
def ingest():
    _run_ingest(reingest=True)


def _run_ingest(reingest=False):
    """
    :param reingest: If `True`, reingest data even if it has already been ingested previously.
                     If `False`, only ingest data even it has not previously been ingested.
    """
    pass


@commands.command(help='split data')
def split():
    _run_ingest(reingest=False)
    _run_make("split")

    print("== Showing summary of input data ==\n")
    _maybe_open("split_summary.html")

    print("Split data into train/test sets")

    print("== Summary of train data ==\n")
    print(pd.read_parquet("split_train.parquet").describe())

    print("== Summary of test data ==\n")
    print(pd.read_parquet("split_test.parquet").describe())


@commands.command(help="Transform features")
def transform():
    _run_ingest(reingest=False)
    _run_make("transform")

    print("== Summary of transformed features ==\n")
    df = pd.read_parquet("transform_train_transformed.parquet")
    X = np.vstack(df["features"])
    print(pd.DataFrame(X).describe())

@commands.command(help='Train a model')
def train():
    _run_ingest(reingest=False)
    _run_make("train")

    print("== Trained a model at train_pipeline.pkl ==\n")

@commands.command(help='evaluate a model (explanations included)')
def evaluate():
    _run_ingest(reingest=False)
    _run_make("evaluate")

    print("== Created the model card ==\n")
    _maybe_open("evaluate_results.html")

@commands.command(help='Clean')
def clean():
    _run_make("clean")

@commands.command(help='Show DAG')
def inspect():
    raise NotImplemented
