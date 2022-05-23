import pathlib
import yaml

import pytest
from click.testing import CliRunner

import mlflow
import mlflow.pipelines
import mlflow.pipelines.cli as pipelines_cli
from mlflow.pipelines import Pipeline

# pylint: disable=unused-import
from tests.pipelines.helper_functions import (
    enter_pipeline_example_directory,
    enter_test_pipeline_directory,
)


@pytest.fixture
def clean_up_pipeline():
    try:
        yield
    finally:
        mlflow.pipelines.clean()


@pytest.mark.usefixtures("enter_pipeline_example_directory", "clean_up_pipeline")
def test_pipelines_api_flow_completes_successfully():
    mlflow.pipelines.clean()
    mlflow.pipelines.evaluate()

    mlflow.pipelines.clean()
    mlflow.pipelines.ingest()
    mlflow.pipelines.split()
    mlflow.pipelines.transform()
    mlflow.pipelines.train()
    mlflow.pipelines.evaluate()


@pytest.mark.usefixtures("enter_pipeline_example_directory", "clean_up_pipeline")
def test_pipelines_cli_flow_completes_successfully():
    CliRunner().invoke(pipelines_cli.clean)
    CliRunner().invoke(pipelines_cli.evaluate)

    CliRunner().invoke(pipelines_cli.clean)
    CliRunner().invoke(pipelines_cli.ingest)
    CliRunner().invoke(pipelines_cli.split)
    CliRunner().invoke(pipelines_cli.transform)
    CliRunner().invoke(pipelines_cli.train)
    CliRunner().invoke(pipelines_cli.evaluate)


@pytest.mark.usefixtures("enter_test_pipeline_directory")
def test_pipelines_log_to_expected_mlflow_backend_and_experiment(tmp_path):
    experiment_name = "my_test_exp"
    tracking_uri = "sqlite:///" + str((tmp_path / "tracking_dst.db").resolve())

    profile_path = pathlib.Path.cwd() / "profiles" / "local.yaml"
    with open(profile_path, "r") as f:
        profile_contents = yaml.safe_load(f)

    profile_contents["experiment"]["name"] = experiment_name
    profile_contents["experiment"]["tracking_uri"] = tracking_uri

    with open(profile_path, "w") as f:
        yaml.safe_dump(profile_contents, f)

    mlflow.pipelines.clean()
    pipeline = Pipeline(profile="local")
    pipeline.evaluate()

    mlflow.set_tracking_uri(tracking_uri)
    logged_runs = mlflow.search_runs(experiment_names=[experiment_name], output_format="list")
    assert len(logged_runs) == 1
    logged_run = logged_runs[0]


@pytest.mark.usefixtures("enter_pipeline_example_directory", "clean_up_pipeline")
@pytest.mark.parametrize("custom_execution_directory", [None, "custom"])
def test_pipelines_execution_directory_is_managed_as_expected(custom_execution_directory, tmp_path):
    if custom_execution_directory is not None:
        custom_execution_directory = tmp_path / custom_execution_directory

    cli_env = {}
    if custom_execution_directory is not None:
        cli_env["MLFLOW_PIPELINES_EXECUTION_DIRECTORY"] = str(custom_execution_directory)

    expected_execution_directory_location = (
        pathlib.Path(custom_execution_directory)
        if custom_execution_directory
        else pathlib.Path.home() / ".mlflow" / "pipelines" / "sklearn_regression"
    )

    # Run the full pipeline and verify that outputs for each step were written to the expected
    # execution directory locations
    CliRunner().invoke(cli=pipelines_cli.evaluate, env=cli_env)
    assert (expected_execution_directory_location / "Makefile").exists()
    assert (expected_execution_directory_location / "steps").exists()
    for step_name in ["ingest", "split", "train", "transform", "evaluate"]:
        step_outputs_path = expected_execution_directory_location / "steps" / step_name / "outputs"
        assert step_outputs_path.exists()
        first_output = next(step_outputs_path.iterdir(), None)
        # TODO: Assert that the ingest step has outputs once ingest execution has been implemented
        assert first_output is not None or step_name == "ingest"

    # Clean the pipeline and verify that all step outputs have been removed
    CliRunner().invoke(cli=pipelines_cli.clean, env=cli_env)
    for step_name in ["ingest", "split", "train", "transform", "evaluate"]:
        step_outputs_path = expected_execution_directory_location / "steps" / step_name / "outputs"
        assert not step_outputs_path.exists()
