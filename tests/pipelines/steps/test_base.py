from unittest import TestCase
from unittest.mock import patch
from mlflow.pipelines.steps.base import BaseStep


class BaseStepImplemented(BaseStep):
    def _run(self, output_directory):
        pass

    def inspect(self, output_directory):
        pass

    def clean(self):
        pass

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        pass

    @property
    def name(self):
        pass


class TestBaseStep(TestCase):
    def test_setup_mlflow_tracking_URI(self):
        URI = "file://file-path"
        with patch("mlflow.set_tracking_uri") as set_tracking_uri:
            step = BaseStepImplemented({BaseStep._TRACKING_URI_CONFIG_KEY: URI}, "")
            step.run("")
            set_tracking_uri.assert_called_once_with(URI)

    def test_from_step_config_path(self):
        step_config_path = "step_config_path"
        pipeline_root = "pipeline_root"
        with patch("mlflow.utils.file_utils.read_yaml") as read_yaml:
            baseStepInstance = BaseStepImplemented.from_step_config_path(
                step_config_path, pipeline_root
            )
            assert baseStepInstance.step_config == read_yaml.return_value
            read_yaml.assert_called_once_with(pipeline_root, step_config_path)
