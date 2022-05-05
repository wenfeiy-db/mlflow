import mlflow

# pylint: disable=unused-import
from tests.pipelines.helper_functions import BaseStepImplemented, enter_pipeline_example_directory


def test_mlflow_tracking_uri_is_set_during_run(enter_pipeline_example_directory):
    pipeline_root_path = enter_pipeline_example_directory
    test_mlflow_uri = "file://file-path"
    step = BaseStepImplemented(
        {BaseStepImplemented._TRACKING_URI_CONFIG_KEY: test_mlflow_uri},
        pipeline_root_path,
    )
    step.run("")
    assert mlflow.get_tracking_uri() == test_mlflow_uri
