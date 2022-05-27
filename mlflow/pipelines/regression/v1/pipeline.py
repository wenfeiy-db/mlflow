import os

from mlflow.pipelines.pipeline import _BasePipeline
from mlflow.pipelines.regression.v1.steps.ingest import IngestStep
from mlflow.pipelines.regression.v1.steps.split import SplitStep
from mlflow.pipelines.regression.v1.steps.transform import TransformStep
from mlflow.pipelines.regression.v1.steps.train import TrainStep
from mlflow.pipelines.regression.v1.steps.evaluate import EvaluateStep
from mlflow.pipelines.regression.v1.steps.register import RegisterStep
from mlflow.pipelines.step import BaseStep
from typing import List
from mlflow.pipelines.utils.execution import get_execution_directory_path


class RegressionPipeline(_BasePipeline):
    _PIPELINE_STEPS = (IngestStep, SplitStep, TransformStep, TrainStep, EvaluateStep, RegisterStep)

    def _get_step_classes(self) -> List[BaseStep]:
        return self._PIPELINE_STEPS

    def _get_pipeline_dag_file(self) -> str:
        # This is just a POC to show how the help strings would be dynamic.
        # TODO: replace below code to read the help strings instead.
        pipeline_yaml = "initial config for the pipeline"

        import jinja2

        j2_env = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.dirname(__file__)))
        pipeline_dag_template = j2_env.get_template("resources/pipeline_dag_template.html").render(
            {"pipeline_yaml": pipeline_yaml}
        )

        pipeline_dag_file = os.path.join(
            get_execution_directory_path(self._name), "pipeline_dag.html"
        )
        with open(pipeline_dag_file, "w") as f:
            f.write(pipeline_dag_template)

        return pipeline_dag_file
