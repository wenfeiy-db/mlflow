import logging
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

_logger = logging.getLogger(__name__)


class RegressionPipeline(_BasePipeline):
    _PIPELINE_STEPS = (
        IngestStep, SplitStep, TransformStep, TrainStep, EvaluateStep, RegisterStep
    )

    def _get_step_classes(self) -> List[BaseStep]:
        return self._PIPELINE_STEPS

    def _get_pipeline_dag_file(self) -> str:
        return os.path.join(os.path.dirname(__file__), "resources/pipeline_dag.html")
