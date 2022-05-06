import pytest

from unittest.mock import patch
from mlflow.pipelines.pipeline import Pipeline
from mlflow.pipelines.regression.v1.pipeline import Pipeline as PipelineV1
from mlflow.exceptions import MlflowException


def test_setup_pipeline_initialization():
    pipeline_root = "pipeline-root"
    profile = "production"
    pipeline_config = {"template": "regression/v1"}

    with patch(
        "mlflow.utils.file_utils.render_and_merge_yaml", return_value=pipeline_config
    ) as patch_render_and_merge_yaml, patch(
        "os.path.join", return_value="profiles/production.yaml"
    ) as patch_os_path_join, patch(
        "mlflow.pipelines.pipeline.get_pipeline_root_path", return_value=pipeline_root
    ) as patch_get_pipeline_root_path:
        pipeline = Pipeline(profile)
        patch_os_path_join.assert_called_once_with("profiles", profile + ".yaml")
        patch_render_and_merge_yaml.assert_called_once_with(
            pipeline_root, "pipeline.yaml", "profiles/production.yaml"
        )
        patch_get_pipeline_root_path.assert_called_once()
        assert type(pipeline) is PipelineV1


def test_error_pipeline_initialization():
    pipeline_root = "pipeline-root"
    profile = "production"
    error_pipeline_template = {"template": "regression/v4"}
    empty_pipeline_template = {}
    patch_get_pipeline_root_path = patch(
        "mlflow.pipelines.pipeline.get_pipeline_root_path", return_value=pipeline_root
    )

    with patch_get_pipeline_root_path, patch(
        "mlflow.utils.file_utils.render_and_merge_yaml", return_value=error_pipeline_template
    ), pytest.raises(MlflowException, match="The template defined in pipeline.yaml is not valid"):
        Pipeline(profile)
        patch_get_pipeline_root_path.assert_called_once()

    with patch_get_pipeline_root_path, patch(
        "mlflow.utils.file_utils.render_and_merge_yaml", return_value=empty_pipeline_template
    ), pytest.raises(
        MlflowException, match="Template property needs to be defined in the pipeline.yaml file"
    ):
        Pipeline(profile)
        patch_get_pipeline_root_path.assert_called_once()
