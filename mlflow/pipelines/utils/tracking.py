import logging

from mlflow.entities import Experiment
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.fluent import set_experiment as fluent_set_experiment

_logger = logging.getLogger(__name__)


def set_experiment(
    experiment_id: str = None, experiment_name: str = None, artifact_location: str = None
) -> Experiment:
    client = MlflowClient()
    if experiment_name is not None:
        experiment = client.get_experiment_by_name(name=experiment_name)
        if not experiment:
            _logger.info(
                "Experiment with name '%s' does not exist. Creating a new experiment.",
                experiment_name,
            )
            client.create_experiment(name=experiment_name, artifact_location=artifact_location)

    return fluent_set_experiment(experiment_id=experiment_id, experiment_name=experiment_name)
