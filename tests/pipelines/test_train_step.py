from pathlib import Path
from typing import Generator

import mlflow
from mlflow.pipelines.regression.v1.pipeline import RegressionPipeline
from mlflow.pipelines.utils.execution import get_step_output_path


# pylint: disable-next=unused-import
from tests.pipelines.helper_functions import enter_pipeline_example_directory


def list_artifacts_recursive(
    tracking_uri: str, run_id: str, path: str = None
) -> Generator[str, None, None]:
    artifacts = mlflow.tracking.MlflowClient(tracking_uri).list_artifacts(run_id, path)
    for artifact in artifacts:
        if artifact.is_dir:
            yield from list_artifacts_recursive(tracking_uri, run_id, artifact.path)
        else:
            yield artifact.path


def test_test_step_logs_step_cards_as_artifacts(enter_pipeline_example_directory: str):
    p = RegressionPipeline(enter_pipeline_example_directory, profile="local")
    for step in ("ingest", "split", "transform", "train"):
        p.run(step)

    local_run_id_path = get_step_output_path(
        pipeline_name=p.name,
        step_name="train",
        relative_path="run_id",
    )
    run_id = Path(local_run_id_path).read_text()
    tracking_uri = p._get_step("train").step_config.get("mlflow_tracking_uri")
    artifacts = set(list_artifacts_recursive(tracking_uri, run_id))
    assert artifacts.issuperset(
        {
            "ingest/card.html",
            "split/card.html",
            # TODO: Uncomment once we update transform and train steps to log a step card.
            # "transform/card.html",
            # "train/card.html",
        }
    )
