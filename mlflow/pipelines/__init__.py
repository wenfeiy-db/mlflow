"""
MLflow Pipelines

Fill out help string later
"""

import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys

import mlflow.utils.databricks_utils as databricks_utils
from mlflow.exceptions import MlflowException
from mlflow.utils.rest_utils import http_request

_logger = logging.getLogger(__name__)


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
    import pandas as pd

    _enter_repository_root()
    _run_ingest(reingest=False)
    _run_make("split")

    _logger.info("== Showing summary of input data ==\n")
    _maybe_open("split_summary.html")

    _logger.info("Split data into train/test sets")

    _logger.info("== Summary of train data ==\n")
    _logger.info(pd.read_parquet("split_train.parquet").describe())

    _logger.info("== Summary of test data ==\n")
    _logger.info(pd.read_parquet("split_test.parquet").describe())


def transform():
    """
    Transform features
    """
    import numpy as np
    import pandas as pd

    _enter_repository_root()
    _run_ingest(reingest=False)
    _run_make("transform")

    _logger.info("== Summary of transformed features ==\n")
    df = pd.read_parquet("transform_train_transformed.parquet")
    X = np.vstack(df["features"])
    _logger.info(pd.DataFrame(X).describe())


def train():
    """
    Train a model
    """
    _enter_repository_root()
    _run_ingest(reingest=False)
    _run_make("train")

    _logger.info("== Trained a model at train_pipeline.pkl ==\n")


def evaluate():
    """
    Evaluate a model (explanations included)
    """
    _enter_repository_root()
    _run_ingest(reingest=False)
    _run_make("evaluate")

    _logger.info("== Created the model card ==\n")
    _maybe_open("evaluate_explanations.html")

    _logger.info("== Produced evaluation metrics ==\n")
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
    for c in iter(lambda: process.stdout.read(1), b""):
        sys.stdout.write(c.decode(sys.stdout.encoding))

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(process.stderr.decode())


def _run_make(rule_name):
    _run_in_subprocess_and_stream_results(["make", rule_name])


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


def _enter_repository_root():
    # In the release version of MLflow Pipelines, each pipeline will be its own git repository.
    # To improve develop. To improve developer velocity for now, we choose to treat a pipeline as
    # a directory, which may be a subdirectory of a git repo. The logic for resolving the
    # repository root for development purposes finds the first `pipeline.yaml` file by traversing
    # up the directory tree, while the release version will find the pipeline repository root
    # (commented out below)
    curr_dir_path = pathlib.Path.cwd()
    root_dir_path = pathlib.Path(curr_dir_path.root)

    while curr_dir_path != root_dir_path:
        pipeline_yaml_path_to_check = curr_dir_path / "pipeline.yaml"
        if pipeline_yaml_path_to_check.exists():
             os.chdir(pipeline_yaml_path_to_check)
             return

    raise MlflowException("Failed to find pipeline.yaml!")

    # TODO: Figure out how to do this in Databricks Jobs (notebook ID isn't available)
    # if databricks_utils.is_in_databricks_repo_notebook():
    #     repo_root = _get_databricks_repo_root_path(databricks_utils.get_notebook_id())
    # else:
    #     # Replace with gitpython later if necessary / possible, since this is
    #     # already an MLflow dependency
    #     repo_root = (
    #         subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
    #         .decode("utf-8")
    #         .rstrip("\n")
    #     )
    #
    # os.chdir(repo_root)


def _get_databricks_repo_root_path(repo_notebook_id):
    OBJECT_GIT_INFORMATION_ENDPOINT = "/api/2.0/workspace/get-object-git-information"

    repo_notebook_git_info_response = http_request(
        host_creds=databricks_utils.get_databricks_host_creds(),
        endpoint=OBJECT_GIT_INFORMATION_ENDPOINT,
        method="GET",
        params={
            "object_id": repo_notebook_id,
        },
    )
    repo_notebook_git_info = json.loads(repo_notebook_git_info_response.text)
    relative_path = repo_notebook_git_info["relative_path"]
    absolute_path = repo_notebook_git_info["absolute_path"]
    # Remove the relative path of the current notebook from the absolute path to obtain the repo
    # root path. Then, prepend `/Workspace` because repos are mounted to the `/Workspaces`
    # directory on the cluster filesystem
    return "/Workspace" + absolute_path[: -1 * len(relative_path)]
