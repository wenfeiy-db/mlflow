import json
import pathlib

import mlflow.utils.databricks_utils as databricks_utils
from mlflow.exceptions import MlflowException
from mlflow.utils.rest_utils import http_request
from mlflow.utils.file_utils import read_yaml


def get_pipeline_name(pipeline_root_path=None):
    pipeline_root_path = pipeline_root_path or get_pipeline_root_path()
    pipeline_config = read_yaml(root=pipeline_root_path, file_name="pipeline.yaml")
    return pipeline_config["name"]


def get_pipeline_root_path():
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
            return str(curr_dir_path)
        else:
            curr_dir_path = curr_dir_path.parent

    raise MlflowException("Failed to find pipeline.yaml!")


def get_databricks_repo_root_path(repo_notebook_id):
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
