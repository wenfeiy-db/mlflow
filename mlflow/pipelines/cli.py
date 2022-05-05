import click
import mlflow.pipelines

from mlflow.pipelines.pipeline import Pipeline
from mlflow.exceptions import MlflowException


@click.group("pipelines")
def commands():
    pass


@commands.command(help="Pipeline initialization")
@click.option(
    "--step",
    type=click.STRING,
    help="Specify the step that needs to run",
    required=True,
)
def run(step):
    # TODO: replace to use env instead of hard coding the profile
    pipeline_module = Pipeline("local")
    try:
        getattr(pipeline_module, step)()
    except AttributeError:
        raise MlflowException("Not a valid step input")


@commands.command(help="Ingest data")
def ingest():
    mlflow.pipelines.ingest()


@commands.command(help="Split data")
def split():
    mlflow.pipelines.split()


@commands.command(help="Transform features")
def transform():
    mlflow.pipelines.transform()


@commands.command(help="Train a model")
def train():
    mlflow.pipelines.train()


@commands.command(help="Evaluate a model (explanations included)")
def evaluate():
    mlflow.pipelines.evaluate()


@commands.command(help="Clean")
def clean():
    mlflow.pipelines.clean()


@commands.command(help="Inspect specific steps or full pipeline DAG")
def inspect():
    mlflow.pipelines.inspect()
