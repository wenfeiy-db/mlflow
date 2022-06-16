import logging
import os
import pathlib
from typing import Dict, Any

from mlflow.exceptions import MlflowException, INVALID_PARAMETER_VALUE
from mlflow.utils.databricks_utils import (
    is_running_in_ipython_environment,
    is_in_databricks_runtime,
)
from mlflow.utils.file_utils import read_yaml, render_and_merge_yaml

_PIPELINE_CONFIG_FILE_NAME = "pipeline.yaml"
_PIPELINE_PROFILE_DIR = "profiles"
_PIPELINE_PROFILE_ENV_VAR = "MLFLOW_PIPELINES_PROFILE"

_logger = logging.getLogger(__name__)


def get_pipeline_name(pipeline_root_path: str = None) -> str:
    """
    Obtains the name of the specified pipeline or of the pipeline corresponding to the current
    working directory.

    :param pipeline_root_path: The absolute path of the pipeline root directory on the local
                               filesystem. If unspecified, the pipeline root directory is
                               resolved from the current working directory.
    :raises MlflowException: If the specified ``pipeline_root_path`` is not a pipeline root
                             directory or if ``pipeline_root_path`` is ``None`` and the current
                             working directory does not correspond to a pipeline.
    :return: The name of the specified pipeline.
    """
    pipeline_root_path = pipeline_root_path or get_pipeline_root_path()
    _verify_is_pipeline_root_directory(pipeline_root_path=pipeline_root_path)
    return os.path.basename(pipeline_root_path)


def get_pipeline_config(pipeline_root_path: str = None, profile: str = None) -> Dict[str, Any]:
    """
    Obtains a dictionary representation of the configuration for the specified pipeline.

    :param pipeline_root_path: The absolute path of the pipeline root directory on the local
                               filesystem. If unspecified, the pipeline root directory is
                               resolved from the current working directory, and an
    :raises MlflowException: If the specified ``pipeline_root_path`` is not a pipeline root
                             directory or if ``pipeline_root_path`` is ``None`` and the current
                             working directory does not correspond to a pipeline.
    :return: The configuration of the specified pipeline.
    """
    pipeline_root_path = pipeline_root_path or get_pipeline_root_path()
    _verify_is_pipeline_root_directory(pipeline_root_path=pipeline_root_path)
    if profile:
        profile_file_name = os.path.join(_PIPELINE_PROFILE_DIR, f"{profile}.yaml")
        return render_and_merge_yaml(
            pipeline_root_path, _PIPELINE_CONFIG_FILE_NAME, profile_file_name
        )
    else:
        return read_yaml(pipeline_root_path, _PIPELINE_CONFIG_FILE_NAME)


def get_pipeline_root_path() -> str:
    """
    Obtains the path of the pipeline corresponding to the current working directory, throwing an
    ``MlflowException`` if the current working directory does not reside within a pipeline
    directory.

    :return: The absolute path of the pipeline root directory on the local filesystem.
    """
    # In the release version of MLflow Pipelines, each pipeline will be its own git repository.
    # To improve developer velocity for now, we choose to treat a pipeline as a directory, which
    # may be a subdirectory of a git repo. The logic for resolving the repository root for
    # development purposes finds the first `pipeline.yaml` file by traversing up the directory
    # tree, while the release version will find the pipeline repository root (commented out below)
    curr_dir_path = pathlib.Path.cwd()

    while True:
        pipeline_yaml_path_to_check = curr_dir_path / _PIPELINE_CONFIG_FILE_NAME
        if pipeline_yaml_path_to_check.exists():
            return str(curr_dir_path.resolve())
        elif curr_dir_path != curr_dir_path.parent:
            curr_dir_path = curr_dir_path.parent
        else:
            # If curr_dir_path == curr_dir_path.parent,
            # we have reached the root directory without finding
            # the desired pipeline.yaml file
            raise MlflowException(f"Failed to find {_PIPELINE_CONFIG_FILE_NAME}!")


def get_default_profile() -> str:
    """
    Returns the default profile name under which a pipeline is executed. The default
    profile may change depending on runtime environment.

    :return: The default profile name string.
    """
    return "databricks" if is_in_databricks_runtime() else "local"


def _verify_is_pipeline_root_directory(pipeline_root_path: str) -> str:
    """
    Verifies that the specified local filesystem path is the path of a pipeline root directory.

    :param pipeline_root_path: The absolute path of the pipeline root directory on the local
                               filesystem to validate.
    :raises MlflowException: If the specified ``pipeline_root_path`` is not a pipeline root
                             directory.
    """
    pipeline_yaml_path = os.path.join(pipeline_root_path, _PIPELINE_CONFIG_FILE_NAME)
    if not os.path.exists(pipeline_yaml_path):
        raise MlflowException(
            f"Failed to find {_PIPELINE_CONFIG_FILE_NAME} in {pipeline_yaml_path}!"
        )


def display_html(html_data: str = None, html_file_path: str = None):
    if html_file_path is None and html_data is None:
        raise MlflowException(
            "At least one HTML source must be provided. html_data and html_file_path are empty.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if is_in_databricks_runtime():
        # Patch IPython display with Databricks display
        import IPython.core.display as icd

        icd.display = display  # pylint: disable=undefined-variable
    if is_running_in_ipython_environment():
        from IPython.display import display as ip_display, HTML

        ip_display(HTML(data=html_data, filename=html_file_path))
    else:
        import shutil
        import subprocess

        if os.path.exists(html_file_path) and shutil.which("open") is not None:
            _logger.info(f"Opening HTML file at: '{html_file_path}'")
            subprocess.run(["open", html_file_path], check=True)