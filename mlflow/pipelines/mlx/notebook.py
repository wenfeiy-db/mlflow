import logging
import subprocess
import os
import shutil
from tabnanny import check
import tempfile

import cloudpickle
import pandas as pd
import numpy as np
from sklearn import set_config

# On Databricks, we will monkey-patch display.
from IPython.core.display import display, HTML

_bazel = "bazelisk"

def _get_workspace_dir():
    """
    Gets current project dir based on where "WORKSPACE" file is.

    TODO: add tests
    """
    dir = os.getcwd()
    while not os.path.exists(os.path.join(dir, "WORKSPACE")):
        dir = os.path.dirname(dir)
    return dir

class MLX:

    def __init__(self):
        self.workspace_dir = _get_workspace_dir()
        logging.debug(f"WORKSPACE dir: {self.workspace_dir}")
        self.working_dir = tempfile.mkdtemp()
        logging.debug(f"Working dir: {self.working_dir}")

    def _sync(self):
        cmds = ["rsync", "-rv", "--ignore-errors", self.workspace_dir + "/", self.working_dir]
        logging.debug("rsync command: " + " ".join(cmds))
        p = subprocess.run(cmds, capture_output=True)
        # On WSFS, rsync cannot sync Databricks notebook file placeholders and return 23.
        if p.returncode != 0 and p.returncode != 23:
            raise RuntimeError(p.stderr.decode())
        
    def _run_bazel_build(self, target):
        p = subprocess.run([_bazel, "build", "--show_progress", "--spawn_strategy=local", target], capture_output=True, cwd=self.working_dir)
        if p.returncode != 0:
            raise RuntimeError(p.stderr.decode())

    def _get_bazel_output_path(self, target):
        return os.path.join(self.working_dir, "bazel-bin", target)

    def split(self):
        self._sync()
        self._run_bazel_build("//:split")

        train_df = pd.read_parquet(self._get_bazel_output_path("train.parquet"))
        test_df = pd.read_parquet(self._get_bazel_output_path("test.parquet"))

        print("Train dataset")
        display(train_df)

        print("Test dataset")
        display(test_df)

    def transform(self):
        self._sync()
        self._run_bazel_build("//:transform")

        df = pd.read_parquet(self._get_bazel_output_path("train_transformed.parquet"))
        X = np.vstack(df["features"])

        print("Transformed features")
        display(pd.DataFrame(X))

    def train(self):
        self._sync()
        self._run_bazel_build("//:train")
        
        pipeline_path = self._get_bazel_output_path("pipeline.pkl")
        with open(pipeline_path, "rb") as f:
            # This might pollute the current Python session,
            # which might cause issues during interactive development.
            pipeline = cloudpickle.load(f)
        
        set_config(display="diagram")
        display(pipeline)

    def evaluate(self):
        self._sync()
        self._run_bazel_build("//:evaluate")

        display(HTML(filename=self._get_bazel_output_path("evaluate.html")))

    def clean(self):
        self._sync()
        subprocess.run([_bazel, "clean"], stderr=subprocess.DEVNULL)

    def help(self):
        self._sync()
        help_path = os.path.join(tempfile.mkdtemp(), "help.html")
        logging.debug(f"DAG path: {help_path}")
        cmd = f'{_bazel} query --noimplicit_deps "//:*" --output graph | sed "s/\/\/://g" | dot -Grankdir=BT -Edir=back -Tsvg > {help_path}'
        logging.debug(f"Run bash command in {self.working_dir}: {cmd}")
        p = subprocess.run(['bash', '-c', cmd], cwd=self.working_dir, capture_output=True)
        if p.returncode != 0:
            raise RuntimeError(p.stderr.decode())
        display(HTML(filename=help_path))
