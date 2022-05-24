import os
import pathlib
from typing import Dict, Any

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking.default_experiment import DEFAULT_EXPERIMENT_ID
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.file_utils import read_yaml

_PIPELINE_CONFIG_FILE_NAME = "pipeline.yaml"


def get_pipeline_name(pipeline_root_path: str = None) -> str:
    """
    Obtains the name of the specified pipeline or of the pipeline corresponding to the current
    working directory.

    :param pipeline_root_path: The absolute path of the pipeline root directory on the local
                               filesystem. If unspecified, the pipeline root directory is
                               resolved from the current working directory, and an
    :raises MlflowException: If the specified ``pipeline_root_path`` is not a pipeline root
                             directory or if ``pipeline_root_path`` is ``None`` and the current
                             working directory does not correspond to a pipeline.
    :return: The name of the specified pipeline.
    """
    pipeline_root_path = pipeline_root_path or get_pipeline_root_path()
    _verify_is_pipeline_root_directory(pipeline_root_path=pipeline_root_path)
    return os.path.basename(pipeline_root_path)


def get_pipeline_config(pipeline_root_path: str = None) -> Dict[str, Any]:
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
    return read_yaml(root=pipeline_root_path, file_name=_PIPELINE_CONFIG_FILE_NAME)


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

    # TODO: Uncomment and fix this pipeline repository root resolution code prior to release,
    #       and replace the current method implementation with this logic
    # TODO: Figure out how to do this in Databricks Jobs (notebook ID isn't available)
    #
    # import json
    # import mlflow.utils.databricks_utils as databricks_utils
    # from mlflow.utils.rest_utils import http_request
    #
    # def _get_databricks_repo_root_path(repo_notebook_id: int) -> str:
    #     """
    #     Obtains the absolute Databricks Workspace Filesystem path to the root directory of the
    #     Databricks Repo associated with the specified Databricks Notebook. This method assumes
    #     that the notebook exists and that it resides in a Databricks Repo.
    #
    #     :param repo_notebook_id: The ID of the Databricks Notebook that resides in a Databricks
    #                              Repo.
    #     :return: The absolute Databricks Workspace Filesystem path to root directory of the
    #              Databricks Repo associated with the specified Databricks Notebook.
    #     """
    #     OBJECT_GIT_INFORMATION_ENDPOINT = "/api/2.0/workspace/get-object-git-information"
    #
    #     repo_notebook_git_info_response = http_request(
    #         host_creds=databricks_utils.get_databricks_host_creds(),
    #         endpoint=OBJECT_GIT_INFORMATION_ENDPOINT,
    #         method="GET",
    #         params={
    #             "object_id": repo_notebook_id,
    #         },
    #     )
    #     repo_notebook_git_info = json.loads(repo_notebook_git_info_response.text)
    #     relative_path = repo_notebook_git_info["relative_path"]
    #     absolute_path = repo_notebook_git_info["absolute_path"]
    #     # Remove the relative path of the current notebook from the absolute path to obtain the
    #     # repo root path. Then, prepend `/Workspace` because repos are mounted to the
    #     # `/Workspaces` directory on the cluster filesystem
    #     return "/Workspace" + absolute_path[: -1 * len(relative_path)]
    #
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


class TrackingConfig:

    _KEY_TRACKING_URI = "mlflow_tracking_uri"
    _KEY_EXPERIMENT_NAME = "mlflow_experiment_name"
    _KEY_EXPERIMENT_ID = "mlflow_experiment_id"
    _KEY_ARTIFACT_LOCATION = "mlflow_experiment_artifact_location"

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str = None,
        experiment_id: str = None,
        artifact_location: str = None,
    ):
        if tracking_uri is None:
            raise MlflowException(
                message="`tracking_uri` must not be None",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if (experiment_name, experiment_id).count(None) != 1:
            raise MlflowException(
                message="Exactly one of `experiment_name` or `experiment_id` must be specified",
                error_code=INVALID_PARAMETER_VALUE,
            )
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id
        self.artifact_location = artifact_location

    def to_dict(self):
        config_dict = {
            TrackingConfig._KEY_TRACKING_URI: self.tracking_uri,
        }

        if self.experiment_name:
            config_dict[TrackingConfig._KEY_EXPERIMENT_NAME] = self.experiment_name

        elif self.experiment_id:
            config_dict[TrackingConfig._KEY_EXPERIMENT_ID] = self.experiment_id

        if self.artifact_location:
            config_dict[TrackingConfig._KEY_ARTIFACT_LOCATION] = self.artifact_location

        return config_dict

    @classmethod
    def from_dict(cls, config_dict):
        return TrackingConfig(
            tracking_uri=config_dict.get(TrackingConfig._KEY_TRACKING_URI),
            experiment_name=config_dict.get(TrackingConfig._KEY_EXPERIMENT_NAME),
            experiment_id=config_dict.get(TrackingConfig._KEY_EXPERIMENT_ID),
            artifact_location=config_dict.get(TrackingConfig._KEY_ARTIFACT_LOCATION),
        )


def get_pipeline_tracking_config(
    pipeline_root_path: str, pipeline_config: Dict[str, Any]
) -> TrackingConfig:
    if is_in_databricks_runtime():
        default_tracking_uri = "databricks"
        default_artifact_location = None
    else:
        mlflow_metadata_base_path = pathlib.Path(pipeline_root_path) / "metadata" / "mlflow"
        default_tracking_sqlite_db_posixpath = (
            (mlflow_metadata_base_path / "mlruns.db").resolve().as_posix()
        )
        default_tracking_uri = f"sqlite:///{default_tracking_sqlite_db_posixpath}"
        default_artifact_location = str((mlflow_metadata_base_path / "mlartifacts").resolve())

    tracking_config = pipeline_config.get("experiment", {})

    config_obj_kwargs = {
        "tracking_uri": tracking_config.get("tracking_uri", default_tracking_uri),
        "artifact_location": tracking_config.get("artifact_location", default_artifact_location),
    }

    experiment_name = tracking_config.get("name")
    if experiment_name is not None:
        return TrackingConfig(
            experiment_name=experiment_name,
            **config_obj_kwargs,
        )

    experiment_id = tracking_config.get("id")
    if experiment_id is not None:
        return TrackingConfig(
            experiment_id=experiment_id,
            **config_obj_kwargs,
        )

    experiment_id = _get_experiment_id()
    if experiment_id != DEFAULT_EXPERIMENT_ID:
        return TrackingConfig(
            experiment_id=experiment_id,
            **config_obj_kwargs,
        )

    return TrackingConfig(
        experiment_name=get_pipeline_name(pipeline_root_path=pipeline_root_path),
        **config_obj_kwargs,
    )


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
        raise MlflowException(f"Failed to find {_PIPELINE_CONFIG_FILE_NAME}!")
