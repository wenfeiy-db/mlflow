import logging

from mlflow.entities import Experiment
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.fluent import set_experiment as fluent_set_experiment

_logger = logging.getLogger(__name__)


def set_experiment(
    experiment_id: str = None, experiment_name: str = None, artifact_location: str = None
) -> Experiment:
    """
    Set the given experiment as the active experiment. The experiment must either be specified by
    name via ``experiment_name`` or by ID via ``experiment_id``. The experiment name and ID cannot
    both be specified.

    :param experiment_name: Case sensitive name of the experiment to be activated. If an experiment
                            with this name does not exist, a new experiment wth this name is
                            created.
    :param experiment_id: ID of the experiment to be activated. If an experiment with this ID
                          does not exist, an exception is thrown.
    :param artifact_location: The optional artifact location to set when creating the experiment,
                              if the experiment does not already exist. If the experiment already
                              exists, ``artifact_location`` is ignored.
    :return: An instance of :py:class:``mlflow.entities.Experiment`` representing the new active
             experiment.
    """
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
