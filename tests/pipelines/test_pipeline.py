import os
import pathlib
import pytest

from mlflow.pipelines.pipeline import Pipeline
from mlflow.exceptions import MlflowException

# pylint: disable=unused-import
from tests.pipelines.helper_functions import (
    enter_pipeline_example_directory,
)  # pylint: enable=unused-import

_STEP_NAMES = ["ingest", "split", "train", "transform", "evaluate"]


@pytest.mark.usefixtures("enter_pipeline_example_directory")
def test_create_pipeline_fails_with_invalid_profile():
    with pytest.raises(MlflowException, match=r".*(Failed to find|does not exist).*"):
        Pipeline(profile="local123")


@pytest.mark.usefixtures("enter_pipeline_example_directory")
def test_create_pipeline_and_clean_works():
    p = Pipeline()
    p.clean()


@pytest.mark.large
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
