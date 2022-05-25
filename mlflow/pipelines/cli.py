import click

from mlflow.pipelines.utils import get_default_profile, _PIPELINE_PROFILE_ENV_VAR
from mlflow.pipelines import Pipeline


@click.group("pipelines")
def commands():
    pass


@commands.command(
    help="Run an individual step in the pipeline. If no step is specified, run all"
    "steps sequentially."
)
@click.option(
    "--step",
    type=click.STRING,
    default=None,
    required=False,
    help="The name of the pipeline step to run.",
)
@click.option(
    "--profile",
    envvar=_PIPELINE_PROFILE_ENV_VAR,
    type=click.STRING,
    default=get_default_profile(),
    required=False,
    help="The profile under which the MLflow pipeline and steps will run.",
)
def run(step, profile):
    Pipeline(profile=profile).run(step)


@commands.command(
    help="Clean the cache associated with an individual step run. If the step is not"
    "specified, clean the entire pipeline cache."
)
@click.option(
    "--step",
    type=click.STRING,
    default=None,
    required=False,
    help="The pipeline step whose execution cached output to be cleaned.",
)
@click.option(
    "--profile",
    envvar=_PIPELINE_PROFILE_ENV_VAR,
    type=click.STRING,
    default=get_default_profile(),
    required=False,
    help="The profile under which the MLflow pipeline and steps will run.",
)
def clean(step, profile):
    Pipeline(profile=profile).clean(step)


@commands.command(
    help="Inspect a step output. If no step is provided, visualize the full pipeline graph."
)
@click.option(
    "--step",
    type=click.STRING,
    default=None,
    required=False,
    help="The pipeline step to be inspected.",
)
@click.option(
    "--profile",
    envvar=_PIPELINE_PROFILE_ENV_VAR,
    type=click.STRING,
    default=get_default_profile(),
    required=False,
    help="The profile under which the MLflow pipeline and steps will run.",
)
def inspect(step, profile):
    Pipeline(profile=profile).inspect(step)
