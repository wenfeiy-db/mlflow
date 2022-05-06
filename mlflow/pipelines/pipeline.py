import os
import mlflow.utils.file_utils

from mlflow.exceptions import MlflowException
from mlflow.utils.class_utils import _get_class_from_string
from mlflow.pipelines.utils import get_pipeline_root_path


class Pipeline:
    def __new__(cls, profile: str):
        """
        Used to create an instance of MLFlow pipeline based on the template
        that is defined in pipeline.yaml of the pipeline root.

        :param profile: String defining the profile name used for constructing
                        pipeline config.
        """
        pipeline_root = get_pipeline_root_path()
        profile_yaml_subpath = os.path.join("profiles", f"{profile}.yaml")
        pipeline_config = mlflow.utils.file_utils.render_and_merge_yaml(
            pipeline_root, "pipeline.yaml", profile_yaml_subpath
        )

        template = pipeline_config.get("template")
        if template is None:
            raise MlflowException("Template property needs to be defined in the pipeline.yaml file")
        moduleTemplate = template.replace("/", ".")
        class_name = f"mlflow.pipelines.{moduleTemplate}.pipeline.Pipeline"

        try:
            pipeline_class_module = _get_class_from_string(class_name)
        except (AttributeError, ModuleNotFoundError):
            raise MlflowException("The template defined in pipeline.yaml is not valid")

        return pipeline_class_module(profile, pipeline_root)
