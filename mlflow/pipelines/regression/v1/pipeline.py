import logging
import os
import mlflow.utils.file_utils

from mlflow.pipelines.regression.v1.steps.ingest import IngestStep
from mlflow.pipelines.regression.v1.steps.split import SplitStep
from mlflow.pipelines.regression.v1.steps.transform import TransformStep
from mlflow.pipelines.regression.v1.steps.train import TrainStep
from mlflow.pipelines.regression.v1.steps.evaluate import EvaluateStep
from mlflow.pipelines.utils.execution import run_pipeline_step
from mlflow.pipelines.utils import get_pipeline_name

_logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, profile: str, pipeline_root: str) -> None:
        """
        Pipeline for the regression v1.

        :param pipeline_root: String file path to the directory where step
                              are defined.
        :param profile: String defining the profile name used for constructing
                        pipeline config.
        """
        self.pipeline_root = pipeline_root
        self.profile = profile

    def resolve_pipeline_steps(self):
        profile_yaml_subpath = os.path.join("profiles", f"{self.profile}.yaml")
        pipeline_config = mlflow.utils.file_utils.render_and_merge_yaml(
            self.pipeline_root, "pipeline.yaml", profile_yaml_subpath
        )
        pipeline_name = get_pipeline_name()

        pipeline_steps = [
            pipeline_class.from_pipeline_config(pipeline_config, self.pipeline_root)
            for pipeline_class in (IngestStep, SplitStep, TransformStep, TrainStep, EvaluateStep)
        ]
        return pipeline_name, pipeline_steps

    def ingest(self) -> None:
        """
        Step to ingest data for running the regression pipeline and store it to a dataframe.
        """
        (
            pipeline_name,
            [ingestStep, splitStep, transformStep, trainStep, evaluateStep],
        ) = self.resolve_pipeline_steps()
        run_pipeline_step(
            self.pipeline_root,
            pipeline_name,
            [ingestStep, splitStep, transformStep, trainStep, evaluateStep],
            ingestStep,
        )
        _logger.info("in ingest step")

    def split(self) -> None:
        """
        Step to split data into training, validation and testing dataframes.
        """
        (
            pipeline_name,
            [ingestStep, splitStep, transformStep, trainStep, evaluateStep],
        ) = self.resolve_pipeline_steps()
        run_pipeline_step(
            self.pipeline_root,
            pipeline_name,
            [ingestStep, splitStep, transformStep, trainStep, evaluateStep],
            splitStep,
        )
        _logger.info("in split step")

    def transform(self) -> None:
        """
        Step to transform the training dataframe suitable for feature engineering.
        These transformations are meant to be part of the model so that they can be applied
        consistently in training and inference.
        """
        (
            pipeline_name,
            [ingestStep, splitStep, transformStep, trainStep, evaluateStep],
        ) = self.resolve_pipeline_steps()
        run_pipeline_step(
            self.pipeline_root,
            pipeline_name,
            [ingestStep, splitStep, transformStep, trainStep, evaluateStep],
            transformStep,
        )
        _logger.info("in transform step")

    def train(self) -> None:
        """
        Step to train a model using the training data set and any serialized transforms.
        """
        (
            pipeline_name,
            [ingestStep, splitStep, transformStep, trainStep, evaluateStep],
        ) = self.resolve_pipeline_steps()
        run_pipeline_step(
            self.pipeline_root,
            pipeline_name,
            [ingestStep, splitStep, transformStep, trainStep, evaluateStep],
            trainStep,
        )
        _logger.info("in train step")

    def evaluate(self) -> None:
        """
        Step to compute quality metrics using the model and the test split.
        """
        (
            pipeline_name,
            [ingestStep, splitStep, transformStep, trainStep, evaluateStep],
        ) = self.resolve_pipeline_steps()
        run_pipeline_step(
            self.pipeline_root,
            pipeline_name,
            [ingestStep, splitStep, transformStep, trainStep, evaluateStep],
            evaluateStep,
        )
        _logger.info("in evaluate step")
