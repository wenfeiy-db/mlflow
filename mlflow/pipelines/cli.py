import subprocess
import os

import click
import pandas as pd
import numpy as np

import mlflow.pipelines


@click.group("pipelines")
def commands():
    pass

@commands.command(help='Ingest data')
def ingest():
    mlflow.pipelines.ingest()


@commands.command(help='Split data')
def split():
    mlflow.pipelines.split()


@commands.command(help="Transform features")
def transform():
    mlflow.pipelines.transform()


@commands.command(help='Train a model')
def train():
    mlflow.pipelines.train()


@commands.command(help='Evaluate a model (explanations included)')
def evaluate():
    mlflow.pipelines.evaluate()


@commands.command(help='Clean')
def clean():
    mlflow.pipelines.clean()


@commands.command(help='Inspect specific steps or full pipeline DAG')
def inspect():
    mlflow.pipelines.inspect()
