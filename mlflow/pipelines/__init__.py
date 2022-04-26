"""
MLflow Pipelines

Fill out help string later
"""

import logging
import os
import shutil
import subprocess

from mlflow.pipelines.utils import get_pipeline_root_path, get_pipeline_name
from mlflow.pipelines.utils.execution_utils import run_step, clean_execution_state

_logger = logging.getLogger(__name__)


def ingest():
    """
    Ingest data
    """
    _run_ingest(reingest=True)


def split():
    """
    Split data
    """
    import pandas as pd

    _run_ingest(reingest=False)
    split_outputs_path = _run_step("split")

    _logger.info("== Showing summary of input data ==\n")
    _maybe_open(os.path.join(split_outputs_path, "summary.html"))

    _logger.info("Split data into train/test sets")

    _logger.info("== Summary of train data ==\n")
    _logger.info(pd.read_parquet(os.path.join(split_outputs_path, "train.parquet")).describe())

    _logger.info("== Summary of test data ==\n")
    _logger.info(pd.read_parquet(os.path.join(split_outputs_path, "test.parquet")).describe())


def transform():
    """
    Transform features
    """
    import numpy as np
    import pandas as pd

    _run_ingest(reingest=False)
    transform_outputs_path = _run_step("transform")

    _logger.info("== Summary of transformed features ==\n")
    df = pd.read_parquet(os.path.join(transform_outputs_path, "train_transformed.parquet"))
    X = np.vstack(df["features"])
    _logger.info(pd.DataFrame(X).describe())


def train():
    """
    Train a model
    """
    _run_ingest(reingest=False)
    train_outputs_path = _run_step("train")
    trained_pipeline_path = os.path.join(train_outputs_path, "pipeline.pkl")
    _logger.info(f"== Trained a model at {trained_pipeline_path} ==\n")


def evaluate():
    """
    Evaluate a model (explanations included)
    """
    _run_ingest(reingest=False)
    evaluate_outputs_path = _run_step("evaluate")

    _logger.info("== Created the model card ==\n")
    _maybe_open(os.path.join(evaluate_outputs_path, "explanations.html"))

    _logger.info("== Produced evaluation metrics ==\n")
    _maybe_open(os.path.join(evaluate_outputs_path, "metrics.json"))


def clean():
    """
    Clean
    """
    pipeline_name = get_pipeline_name()
    clean_execution_state(pipeline_name=pipeline_name)


def inspect():
    """
    Inspect specific steps or full pipeline DAG
    """
    raise NotImplementedError


def _run_step(step_name):
    """
    Runs the specified step in the current pipeline, where the current pipeline is determined by
    the current working directory.

    :param target_step: The name of the step to run.
    :return: The absolute path of the step's execution outputs on the local filesystem.
    """
    pipeline_root_path = get_pipeline_root_path()
    pipeline_name = get_pipeline_name(pipeline_root_path=pipeline_root_path)
    return run_step(
        pipeline_root_path=pipeline_root_path,
        pipeline_name=pipeline_name,
        pipeline_steps=["ingest", "split", "transform", "train", "evaluate"],
        target_step=step_name,
    )


def _maybe_open(path):
    assert os.path.exists(path), f"{path} does not exist"
    if shutil.which("open") is not None:
        subprocess.run(["open", path], check=True)
    else:
        _logger.info(f"Please open {path} manually.")


def _run_ingest(reingest=False):  # pylint: disable=unused-argument
    """
    :param reingest: If `True`, reingest data even if it has already been ingested previously.
                     If `False`, only ingest data even it has not previously been ingested.
    """
    pass
