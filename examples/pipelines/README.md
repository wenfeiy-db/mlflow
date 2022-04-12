# MLX Demo

## Setup

### Install bazelisk and graphviz

```
brew install bazelisk
brew install graphviz
```

### Install MLX demo

Use conda or virtualenv to create and activate a Python 3.8 environment.
Then run the following:

```
git clone https://github.com/databricks/mlx.git
cd mlx
pip install .
```

### Play with the demo

#### Databricks

[Sync](https://docs.databricks.com/repos.html) this repo and run `notebooks/databricks` on an MLR 10.3 cluster with [workspace files support enabled](https://docs.databricks.com/repos.html#work-with-non-notebook-files-in-a-databricks-repo).

#### Jupyter

Run `notebooks/jupyter.ipynb` under the current Python environment.

#### CLI

```
cd demo
mlx --help
mlx dag
mlx split
mlx transform
mlx train
mlx evaluate
```

Check MLflow UI

```
cd /tmp/mlruns
mlflow ui
```

Modify `train.py` and run

```
mlx evaluate
```

## Apparent gaps

* The `autos.yaml` is not actually used.
* The mlflow experiment folder is hardcoded to `file:/tmp/mlruns`.
* MLflow integration doesn't work on Databricks.
