import sys
import shutil
import subprocess
import tempfile
import os

import click
import pandas as pd
import numpy as np

from mlx.preprocess_step import preprocess_step
from mlx.transform_step import transform_step
from mlx.train_step import train_step
from mlx.evaluate_step import evaluate_step

_bazel = "bazelisk"

def _run_in_subprocess_and_stream_results(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    for c in iter(lambda: process.stdout.read(1), b''):
        sys.stdout.buffer.write(c)

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(process.stderr.decode())

def _run_make(rule_name):
    _run_in_subprocess_and_stream_results(["make", rule_name])
    # subprocess.run(["make", rule_name], capture_output=True, check=True)

def _run_bazel_build(target):
    p = subprocess.run([_bazel, "build", "--spawn_strategy=local", target], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode())

def _maybe_open(path):
    assert os.path.exists(path), f"{path} does not exist"
    if shutil.which("open") is not None:
        subprocess.run(["open", path])
    else:
        print(f"Please open {path} manually.")

@click.group()
def cli():
    pass

cli.add_command(preprocess_step)

cli.add_command(transform_step)

cli.add_command(train_step)

cli.add_command(evaluate_step)

@cli.command(help='Preprocess input data')
def preprocess():
    # _run_bazel_build("//:preprocess")
    _run_make("preprocess")

    print("== Showing summary of input data ==\n")
    # _maybe_open("bazel-bin/summary.html")
    _maybe_open("preprocess_summary.html")

    print("Split data into train/test sets")

    print("== Summary of train data ==\n")
    print(pd.read_parquet("preprocess_train.parquet").describe())

    print("== Summary of test data ==\n")
    print(pd.read_parquet("preprocess_test.parquet").describe())


@cli.command(help="Transform features")
def transform():
    # _run_bazel_build("//:transform")
    _run_make("transform")
    print("== Summary of transformed features ==\n")
    # df = pd.read_parquet("bazel-bin/train_transformed.parquet")
    df = pd.read_parquet("transform_train_transformed.parquet")
    X = np.vstack(df["features"])
    print(pd.DataFrame(X).describe())

@cli.command(help='Train a model')
def train():
    # _run_bazel_build("//:train")
    _run_make("train")
    # print("== Trained a model at pipeline.pkl ==\n")
    print("== Trained a model at train_pipeline.pkl ==\n")

@cli.command(help='evaluate a model')
def evaluate():
    # _run_bazel_build("//:evaluate")
    _run_make("evaluate")
    print("== Created the model card ==\n")
    # _maybe_open("bazel-bin/evaluate_explanations.html")
    _maybe_open("evaluate_results.html")

@cli.command(help='Clean')
def clean():
    _run_make("clean")
    # subprocess.run([_bazel, "clean"], check=True)

@cli.command(help='Show DAG')
def dag():
    # TODO: Generate DAG from MLX logical pipeline, not bazel query, since the pipeline depends on
    # extracted configs & other internal scripts that shouldn't be exposed to the user for clarity
    raise NotImplemented

    # dag_path = os.path.join(tempfile.mkdtemp(), "dag.html")
    # subprocess.run(['bash', '-c', f'{_bazel} query --noimplicit_deps "//:*" --output graph | sed "s/\/\/://g" | dot -Grankdir=RL -Edir=back -Tsvg > {dag_path}'], check=True)
    # _maybe_open(dag_path)

if __name__ == '__main__':
    cli()
