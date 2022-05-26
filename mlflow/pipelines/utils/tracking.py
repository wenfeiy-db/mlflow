import logging
import pathlib
from typing import Dict, Any, TypeVar

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.pipelines.utils import get_pipeline_name
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.default_experiment import DEFAULT_EXPERIMENT_ID
from mlflow.tracking.fluent import set_experiment as fluent_set_experiment, _get_experiment_id
from mlflow.utils.databricks_utils import is_in_databricks_runtime

_logger = logging.getLogger(__name__)


TrackingConfigType = TypeVar("TrackingConfig")


class TrackingConfig:
    """
    The MLflow Tracking configuration associated with an MLflow Pipeline, including the
    Tracking URI and information about the destination Experiment for writing results.
    """

    _KEY_TRACKING_URI = "mlflow_tracking_uri"
    _KEY_EXPERIMENT_NAME = "mlflow_experiment_name"
    _KEY_EXPERIMENT_ID = "mlflow_experiment_id"
    _KEY_ARTIFACT_LOCATION = "mlflow_experiment_artifact_location"

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str = None,
        experiment_id: str = None,
        artifact_location: str = None,
    ):
        """
        :param tracking_uri: The MLflow Tracking URI.
        :param experiment_name: The MLflow Experiment name. At least one of ``experiment_name`` or
                                ``experiment_id`` must be specified. If both are specified, they
                                must be consistent with Tracking server state. Note that this
                                Experiment may not exist prior to pipeline execution.
        :param experiment_id: The MLflow Experiment ID. At least one of ``experiment_name`` or
                              ``experiment_id`` must be specified. If both are specified, they
                              must be consistent with Tracking server state. Note that this
                              Experiment may not exist prior to pipeline execution.
        :param artifact_location: The artifact location to use for the Experiment, if the Experiment
                                  does not already exist. If the Experiment already exists, this
                                  location is ignored.
        """
        if tracking_uri is None:
            raise MlflowException(
                message="`tracking_uri` must be specified",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if (experiment_name, experiment_id).count(None) != 1:
            raise MlflowException(
                message="Exactly one of `experiment_name` or `experiment_id` must be specified",
                error_code=INVALID_PARAMETER_VALUE,
            )
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id
        self.artifact_location = artifact_location

    def to_dict(self) -> Dict[str, str]:
        """
        Obtains a dictionary representation of the MLflow Tracking configuration.

        :return: A dictionary representation of the MLflow Tracking configuration.
        """
        config_dict = {
            TrackingConfig._KEY_TRACKING_URI: self.tracking_uri,
        }

        if self.experiment_name:
            config_dict[TrackingConfig._KEY_EXPERIMENT_NAME] = self.experiment_name

        elif self.experiment_id:
            config_dict[TrackingConfig._KEY_EXPERIMENT_ID] = self.experiment_id

        if self.artifact_location:
            config_dict[TrackingConfig._KEY_ARTIFACT_LOCATION] = self.artifact_location

        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, str]) -> TrackingConfigType:
        """
        Creates a ``TrackingConfig`` instance from a dictionary representation.

        :param config_dict: A dictionary representation of the MLflow Tracking configuration.
        :return: A ``TrackingConfig`` instance.
        """
        return TrackingConfig(
            tracking_uri=config_dict.get(TrackingConfig._KEY_TRACKING_URI),
            experiment_name=config_dict.get(TrackingConfig._KEY_EXPERIMENT_NAME),
            experiment_id=config_dict.get(TrackingConfig._KEY_EXPERIMENT_ID),
            artifact_location=config_dict.get(TrackingConfig._KEY_ARTIFACT_LOCATION),
        )


def get_pipeline_tracking_config(
    pipeline_root_path: str, pipeline_config: Dict[str, Any]
) -> TrackingConfig:
    """
    Obtains the MLflow Tracking configuration for the specified pipeline.

    :param pipeline_root_path: The absolute path of the pipeline root directory on the local
                               filesystem.
    :param pipeline_config: The configuration of the specified pipeline.
    :return: A ``TrackingConfig`` instance containing MLflow Tracking information for the
             specified pipeline, including Tracking URI, Experiment name, and more.
    """
    if is_in_databricks_runtime():
        default_tracking_uri = "databricks"
        default_artifact_location = None
    else:
        mlflow_metadata_base_path = pathlib.Path(pipeline_root_path) / "metadata" / "mlflow"
        mlflow_metadata_base_path.mkdir(exist_ok=True, parents=True)
        default_tracking_sqlite_db_posixpath = (
            (mlflow_metadata_base_path / "mlruns.db").resolve().as_posix()
        )
        default_tracking_uri = f"sqlite:///{default_tracking_sqlite_db_posixpath}"
        default_artifact_location = str((mlflow_metadata_base_path / "mlartifacts").resolve())

    tracking_config = pipeline_config.get("experiment", {})

    config_obj_kwargs = {
        "tracking_uri": tracking_config.get("tracking_uri", default_tracking_uri),
        "artifact_location": tracking_config.get("artifact_location", default_artifact_location),
    }

    experiment_name = tracking_config.get("name")
    if experiment_name is not None:
        return TrackingConfig(
            experiment_name=experiment_name,
            **config_obj_kwargs,
        )

    experiment_id = tracking_config.get("id")
    if experiment_id is not None:
        return TrackingConfig(
            experiment_id=experiment_id,
            **config_obj_kwargs,
        )

    experiment_id = _get_experiment_id()
    if experiment_id != DEFAULT_EXPERIMENT_ID:
        return TrackingConfig(
            experiment_id=experiment_id,
            **config_obj_kwargs,
        )

    return TrackingConfig(
        experiment_name=get_pipeline_name(pipeline_root_path=pipeline_root_path),
        **config_obj_kwargs,
    )


def apply_pipeline_tracking_config(tracking_config: TrackingConfig):
    """
    Applies the specified ``TrackingConfig`` in the current context by setting the associated
    MLflow Tracking URI (via ``mlflow.set_tracking_uri()``) and setting the associated MLflow
    Experiment (via ``mlflow.set_experiment()``), creating it if necessary.

    :param tracking_config: The MLflow Pipeline ``TrackingConfig`` to apply.
    """
    mlflow.set_tracking_uri(uri=tracking_config.tracking_uri)

    client = MlflowClient()
    if tracking_config.experiment_name is not None:
        experiment = client.get_experiment_by_name(name=tracking_config.experiment_name)
        if not experiment:
            _logger.info(
                "Experiment with name '%s' does not exist. Creating a new experiment.",
                tracking_config.experiment_name,
            )
            client.create_experiment(
                name=tracking_config.experiment_name,
                artifact_location=tracking_config.artifact_location,
            )

    fluent_set_experiment(
        experiment_id=tracking_config.experiment_id, experiment_name=tracking_config.experiment_name
    )
