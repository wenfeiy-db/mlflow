import os
import logging

import pandas as pd

import mlflow.pyfunc
from mlflow.tracking.client import MlflowClient
from mlflow.pipelines.pipeline import _BasePipeline
from mlflow.pipelines.regression.v1.steps.ingest import IngestStep
from mlflow.pipelines.regression.v1.steps.split import (
    SplitStep,
    _OUTPUT_TRAIN_FILE_NAME,
    _OUTPUT_VALIDATION_FILE_NAME,
    _OUTPUT_TEST_FILE_NAME,
)
from mlflow.pipelines.regression.v1.steps.transform import TransformStep
from mlflow.pipelines.regression.v1.steps.train import TrainStep
from mlflow.pipelines.regression.v1.steps.evaluate import EvaluateStep
from mlflow.pipelines.regression.v1.steps.register import RegisterStep
from mlflow.pipelines.step import BaseStep
from typing import List
from mlflow.pipelines.utils.execution import get_execution_directory_path
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.exceptions import MlflowException, INVALID_PARAMETER_VALUE
from mlflow.tracking._tracking_service.utils import _use_tracking_uri

_logger = logging.getLogger(__name__)


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

    def get_artifact(self, artifact: str):
        """
        Read an artifact from pipeline output. artifact names can be obtained from
        `Pipeline.inspect()` or `Pipeline.run()` output.

        Returns None if the specified artifact is not found.
        Raise an error if the artifact is not supported.

        :param artifact: A string representing the artifact, supported artifact values are:

         - `ingested_data`: returns the ingested data created in the ingest step as a pandas
           DataFrame.

         - `training_data`: returns the training data created in the split step as a
           pandas DataFrame.

         - `validation_data`: returns the validation data created in the split step as
           a pandas DataFrame.

         - `test_data`: returns the test data created in the split step as a pandas
           DataFrame.

         - `transformed_training_data`: returns the transformed training data created in the
           transform step as a pandas DataFrame.

         - `transformed_validation_data`: returns the transformed validation data created in
           the transform step as a pandas DataFrame.

         - `model`: returns the Pyfunc model from the train step output.

         - `transformer`: returns the sklearn transformer from transform step output.

         - `run`: returns an MLflow run object.
        """
        ingest_step, split_step, transform_step, train_step, _, _ = self._steps

        ingest_output_dir = get_step_output_path(self.name, ingest_step.name, "")
        split_output_dir = get_step_output_path(self.name, split_step.name, "")
        transform_output_dir = get_step_output_path(self.name, transform_step.name, "")
        train_output_dir = get_step_output_path(self.name, train_step.name, "")

        def log_artifact_not_found_warning(artifact_name, step_name):
            _logger.warning(
                f"{artifact_name} is not found. Re-run the {step_name} step to generate."
            )

        def read_run_id():
            run_id_file_path = os.path.join(train_output_dir, "run_id")
            if os.path.exists(run_id_file_path):
                with open(run_id_file_path, "r") as f:
                    return f.read().strip()
            else:
                return None

        train_step_tracking_uri = train_step.tracking_config.tracking_uri

        def read_dataframe(artifact_name, output_dir, file_name, step_name):
            data_path = os.path.join(output_dir, file_name)
            if os.path.exists(data_path):
                return pd.read_parquet(data_path)
            else:
                log_artifact_not_found_warning(artifact_name, step_name)
                return None

        if artifact == "ingested_data":
            return read_dataframe(
                "ingested_data",
                ingest_output_dir,
                IngestStep._DATASET_OUTPUT_NAME,
                ingest_step.name,
            )

        elif artifact == "training_data":
            return read_dataframe(
                "training_data", split_output_dir, _OUTPUT_TRAIN_FILE_NAME, split_step.name
            )

        elif artifact == "validation_data":
            return read_dataframe(
                "validation_data", split_output_dir, _OUTPUT_VALIDATION_FILE_NAME, split_step.name
            )

        elif artifact == "test_data":
            return read_dataframe(
                "test_data", split_output_dir, _OUTPUT_TEST_FILE_NAME, split_step.name
            )

        elif artifact == "transformed_training_data":
            return read_dataframe(
                "transformed_training_data",
                transform_output_dir,
                "transformed_training_data.parquet",
                transform_step.name,
            )

        elif artifact == "transformed_validation_data":
            return read_dataframe(
                "transformed_validation_data",
                transform_output_dir,
                "transformed_validation_data.parquet",
                transform_step.name,
            )

        elif artifact == "model":
            run_id = read_run_id()
            if run_id:
                with _use_tracking_uri(train_step_tracking_uri):
                    return mlflow.pyfunc.load_model(f"runs:/{run_id}/{train_step.name}/model")
            else:
                log_artifact_not_found_warning("model", train_step.name)
                return None

        elif artifact == "transformer":
            run_id = read_run_id()
            if run_id:
                with _use_tracking_uri(train_step_tracking_uri):
                    return mlflow.sklearn.load_model(
                        f"runs:/{run_id}/{transform_step.name}/transformer"
                    )
            else:
                log_artifact_not_found_warning("transformer", train_step.name)
                return None

        elif artifact == "run":
            run_id = read_run_id()
            if run_id:
                with _use_tracking_uri(train_step_tracking_uri):
                    return MlflowClient().get_run(run_id)
            else:
                log_artifact_not_found_warning("mlflow run", train_step.name)
                return None

        else:
            raise MlflowException(
                f"The artifact {artifact} is not supported.", error_code=INVALID_PARAMETER_VALUE
            )
