import os

from unittest import TestCase
from unittest.mock import patch
from mlflow.pipelines.step import BaseStep


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
