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
            step = BaseStepImplemented({"tracking_URI": URI}, "")
            step.run("")
            set_tracking_uri.assert_called_once_with(URI)
