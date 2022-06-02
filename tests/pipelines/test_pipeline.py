import os
import pathlib
import pytest
import yaml
from typing import Generator

import mlflow
from mlflow.pipelines.pipeline import Pipeline
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.exceptions import MlflowException
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.context.registry import resolve_tags
from mlflow.utils.file_utils import path_to_local_file_uri

# pylint: disable=unused-import
from tests.pipelines.helper_functions import (
    enter_pipeline_example_directory,
    enter_test_pipeline_directory,
)  # pylint: enable=unused-import

_STEP_NAMES = ["ingest", "split", "train", "transform", "evaluate"]


def list_all_artifacts(
    tracking_uri: str, run_id: str, path: str = None
) -> Generator[str, None, None]:
    artifacts = mlflow.tracking.MlflowClient(tracking_uri).list_artifacts(run_id, path)
    for artifact in artifacts:
        if artifact.is_dir:
            yield from list_all_artifacts(tracking_uri, run_id, artifact.path)
        else:
            yield artifact.path


@pytest.mark.usefixtures("enter_pipeline_example_directory")
def test_create_pipeline_fails_with_invalid_profile():
    with pytest.raises(MlflowException, match=r".*(Failed to find|does not exist).*"):
        Pipeline(profile="local123")


@pytest.mark.usefixtures("enter_pipeline_example_directory")
def test_create_pipeline_and_clean_works():
    p = Pipeline()
    p.clean()


@pytest.mark.usefixtures("enter_pipeline_example_directory")
@pytest.mark.parametrize("custom_execution_directory", [None, "custom"])
def test_pipelines_execution_directory_is_managed_as_expected(custom_execution_directory, tmp_path):
    if custom_execution_directory is not None:
        custom_execution_directory = tmp_path / custom_execution_directory

    if custom_execution_directory is not None:
        os.environ["MLFLOW_PIPELINES_EXECUTION_DIRECTORY"] = str(custom_execution_directory)

    expected_execution_directory_location = (
        pathlib.Path(custom_execution_directory)
        if custom_execution_directory
        else pathlib.Path.home() / ".mlflow" / "pipelines" / "sklearn_regression"
    )

    # Run the full pipeline and verify that outputs for each step were written to the expected
    # execution directory locations
    p = Pipeline()
    p.run()
    assert (expected_execution_directory_location / "Makefile").exists()
    assert (expected_execution_directory_location / "steps").exists()
    for step_name in _STEP_NAMES:
        step_outputs_path = expected_execution_directory_location / "steps" / step_name / "outputs"
        assert step_outputs_path.exists()
        first_output = next(step_outputs_path.iterdir(), None)
        # TODO: Assert that the ingest step has outputs once ingest execution has been implemented
        assert first_output is not None or step_name == "ingest"

    # Clean the pipeline and verify that all step outputs have been removed
    p.clean()
    for step_name in _STEP_NAMES:
        step_outputs_path = expected_execution_directory_location / "steps" / step_name / "outputs"
        assert not step_outputs_path.exists()


@pytest.mark.usefixtures("enter_test_pipeline_directory")
def test_pipelines_log_to_expected_mlflow_backend_and_experiment_with_expected_run_tags(tmp_path):
    experiment_name = "my_test_exp"
    tracking_uri = "sqlite:///" + str((tmp_path / "tracking_dst.db").resolve())
    artifact_location = str((tmp_path / "mlartifacts").resolve())

    profile_path = pathlib.Path.cwd() / "profiles" / "local.yaml"
    with open(profile_path, "r") as f:
        profile_contents = yaml.safe_load(f)

    profile_contents["experiment"]["name"] = experiment_name
    profile_contents["experiment"]["tracking_uri"] = tracking_uri
    profile_contents["experiment"]["artifact_location"] = path_to_local_file_uri(artifact_location)

    with open(profile_path, "w") as f:
        yaml.safe_dump(profile_contents, f)

    pipeline = Pipeline(profile="local")
    pipeline.clean()
    pipeline.run()

    mlflow.set_tracking_uri(tracking_uri)
    logged_runs = mlflow.search_runs(experiment_names=[experiment_name], output_format="list")
    assert len(logged_runs) == 1
    logged_run = logged_runs[0]
    assert logged_run.info.artifact_uri == path_to_local_file_uri(
        str((pathlib.Path(artifact_location) / logged_run.info.run_id / "artifacts").resolve())
    )
    assert "r2_score_on_data_test" in logged_run.data.metrics
    artifacts = MlflowClient(tracking_uri).list_artifacts(run_id=logged_run.info.run_id)
    assert "model" in [artifact.path for artifact in artifacts]
    run_tags = MlflowClient(tracking_uri).get_run(run_id=logged_run.info.run_id).data.tags
    assert resolve_tags().items() <= run_tags.items()


@pytest.mark.usefixtures("enter_pipeline_example_directory")
def test_test_step_logs_step_cards_as_artifacts():
    p = Pipeline()
    p.run("ingest")
    p.run("split")
    p.run("transform")
    p.run("train")

    local_run_id_path = get_step_output_path(
        pipeline_name=p.name,
        step_name="train",
        relative_path="run_id",
    )
    run_id = pathlib.Path(local_run_id_path).read_text()
    tracking_uri = p._get_step("train").step_config.get("mlflow_tracking_uri")
    artifacts = set(list_all_artifacts(tracking_uri, run_id))
    assert artifacts.issuperset(
        {
            "ingest/card.html",
            "split/card.html",
            # TODO: Uncomment once we update transform and train steps to log a step card.
            # "transform/card.html",
            # "train/card.html",
        }
    )
