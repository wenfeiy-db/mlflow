"""
MLflow Pipelines

Fill out help string later
"""

import logging
import os
import shutil
import subprocess
from typing import Dict, List, Any

from mlflow.pipelines.utils import get_pipeline_root_path, get_pipeline_name, get_pipeline_config
from mlflow.pipelines.utils.execution import run_pipeline_step, clean_execution_state
from mlflow.pipelines.regression.v1.steps.ingest import IngestStep
from mlflow.pipelines.regression.v1.steps.split import SplitStep
from mlflow.pipelines.regression.v1.steps.transform import TransformStep
from mlflow.pipelines.regression.v1.steps.train import TrainStep
from mlflow.pipelines.regression.v1.steps.evaluate import EvaluateStep
from mlflow.pipelines.step import BaseStep 

_logger = logging.getLogger(__name__)


def ingest():
    """
    Ingest data
    """
    _run_pipeline_step("ingest")


def split():
    """
    Split data
    """
    import pandas as pd

    split_outputs_path = _run_pipeline_step("split")

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

    transform_outputs_path = _run_pipeline_step("transform")

    _logger.info("== Summary of transformed features ==\n")
    df = pd.read_parquet(os.path.join(transform_outputs_path, "train_transformed.parquet"))
    X = np.vstack(df["features"])
    _logger.info(pd.DataFrame(X).describe())


def train():
    """
    Train a model
    """
    train_outputs_path = _run_pipeline_step("train")
    trained_pipeline_path = os.path.join(train_outputs_path, "pipeline.pkl")
    _logger.info(f"== Trained a model at {trained_pipeline_path} ==\n")


def evaluate():
    """
    Evaluate a model (explanations included)
    """
    evaluate_outputs_path = _run_pipeline_step("evaluate")

    _logger.info("== Created the model card ==\n")
    _maybe_open(os.path.join(evaluate_outputs_path, "explanations.html"))

    _logger.info("== Produced evaluation metrics ==\n")
    _maybe_open(os.path.join(evaluate_outputs_path, "metrics.json"))


def clean():
    """
    Clean
    """
    pipeline_root_path = get_pipeline_root_path()
    pipeline_config = get_pipeline_config(pipeline_root_path=pipeline_root_path)
    pipeline_name = get_pipeline_name()
    pipeline_steps = _get_pipeline_steps(
        pipeline_root_path=pipeline_root_path,
        pipeline_config=pipeline_config,
    )
    clean_execution_state(pipeline_name=pipeline_name, pipeline_steps=pipeline_steps)


def inspect():
    """
    Inspect specific steps or full pipeline DAG
    """
    raise NotImplementedError


def _run_pipeline_step(step_name: str) -> str:
    """
    Runs the specified step in the current pipeline, where the current pipeline is determined by
    the current working directory.

    :param target_step: The name of the step to run.
    :return: The absolute path of the step's execution outputs on the local filesystem.
    """
    pipeline_root_path = get_pipeline_root_path()
    pipeline_config = get_pipeline_config(pipeline_root_path=pipeline_root_path)
    pipeline_name = get_pipeline_name(pipeline_root_path=pipeline_root_path)
    pipeline_steps = _get_pipeline_steps(
        pipeline_root_path=pipeline_root_path,
        pipeline_config=pipeline_config,
    )
    return run_pipeline_step(
        pipeline_root_path=pipeline_root_path,
        pipeline_name=pipeline_name,
        pipeline_steps=pipeline_steps,
        target_step=[step for step in pipeline_steps if step.name == step_name][0],
    )


def _get_pipeline_steps(pipeline_root_path: str, pipeline_config: Dict[str, Any]) -> List[BaseStep]:
    """
    :param pipeline_root_path: The absolute path of the pipeline root directory on the local
                               filesystem.
    :param pipeline_config: The configuration of the specified pipeline.
    :return: A list of all steps contained in the pipeline, where each step occurs after the
             previous step.
    """
    return [
        pipeline_class.from_pipeline_config(
            pipeline_config=pipeline_config,
            pipeline_root=pipeline_root_path,
        )
        for pipeline_class in (IngestStep, SplitStep, TransformStep, TrainStep, EvaluateStep)
    ]


def _maybe_open(path):
    assert os.path.exists(path), f"{path} does not exist"
    if shutil.which("open") is not None:
        subprocess.run(["open", path], check=True)
    else:
        _logger.info(f"Please open {path} manually.")
