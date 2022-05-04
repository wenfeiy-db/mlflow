import json
import logging
import os

import cloudpickle

import mlflow
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.utils.execution import get_step_output_path

_logger = logging.getLogger(__name__)


class EvaluateStep(BaseStep):
    def _run(self, output_directory):
        import pandas as pd

        pipeline_path = get_step_output_path(
            pipeline_name=self.pipeline_name,
            step_name="train",
            relative_path="pipeline.pkl",
        )
        with open(pipeline_path, "rb") as f:
            pipeline = cloudpickle.load(f)

        train_data_path = get_step_output_path(
            pipeline_name=self.pipeline_name,
            step_name="split",
            relative_path="train.parquet",
        )
        test_data_path = get_step_output_path(
            pipeline_name=self.pipeline_name,
            step_name="split",
            relative_path="test.parquet",
        )
        train_data = pd.read_parquet(train_data_path)
        test_data = pd.read_parquet(test_data_path)
        X_train = train_data.drop(columns=["price"])

        run_id_path = get_step_output_path(
            pipeline_name=self.pipeline_name,
            step_name="train",
            relative_path="run_id",
        )
        with open(run_id_path, "r") as f:
            run_id = f.read()

        mlflow.set_experiment("demo")  # hardcoded

        with mlflow.start_run(run_id=run_id):
            EvaluateStep._explain(pipeline, X_train, output_directory)
            EvaluateStep._evaluate(pipeline, train_data, test_data, output_directory)

    @staticmethod
    def _explain(pipeline, X_train, output_directory):
        """
        :param pipeline: The [<transformer>, <trained_model>] pipeline
        :param X_train: Features from the training dataset
        :param output_directory: Path to the output directory for the evaluate step on the local
                                 filesystem.
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import shap
        from matplotlib import rcParams

        mode = X_train.mode().iloc[0]

        background = shap.sample(X_train, 10, random_state=3).fillna(mode)
        sample = shap.sample(X_train, 10, random_state=12).fillna(mode)

        predict = lambda x: pipeline.predict(pd.DataFrame(x, columns=X_train.columns))
        evaluateer = shap.KernelExplainer(predict, background, link="identity")
        shap_values = evaluateer.shap_values(sample)

        # https://giters.com/slundberg/shap/issues/1916
        rcParams.update({"figure.autolayout": True})

        explanations_output_path = os.path.join(output_directory, "explanations.html")
        shap.summary_plot(shap_values, sample, show=False)
        plt.savefig(explanations_output_path, format="svg")

        mlflow.log_artifact(explanations_output_path)

    @staticmethod
    def _evaluate(pipeline, train_data, test_data, output_directory):
        """
        :param pipeline: The [<transformer>, <trained_model>] pipeline
        :param train_data: The training dataset
        :param test_data: The test dataset
        :param output_directory: Path to the output directory for the evaluate step on the local
                                 filesystem.
        """
        train_rmse, train_worst = EvaluateStep._evaluate_model_on_dataset(pipeline, train_data)
        test_rmse, _ = EvaluateStep._evaluate_model_on_dataset(pipeline, test_data)

        metrics = [
            {"dataset": "train", "metric": "rmse", "value": train_rmse},
            {"dataset": "test", "metric": "rmse", "value": test_rmse},
        ]

        with open(os.path.join(output_directory, "metrics.json"), "w") as f:
            json.dump(metrics, f)

        train_worst.to_parquet(os.path.join(output_directory, "worst_training_examples.parquet"))

    @staticmethod
    def _evaluate_model_on_dataset(model, df):
        from sklearn.metrics import mean_squared_error

        # TODO: read from conf
        label_col = "price"
        y_true = df[label_col]
        y_pred = model.predict(df.drop(columns=[label_col]))
        rmse = mean_squared_error(y_true, y_pred, squared=False)

        df["_pred_"] = y_pred
        df["_error_"] = (y_true - y_pred).abs()
        worst = df.nlargest(20, columns=["_error_"])

        return rmse, worst

    def inspect(self, output_directory):
        # Do step-specific code to inspect/materialize the output of the step
        _logger.info("evaluate inspect code %s", output_directory)
        pass

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        step_config = {EvaluateStep._TRACKING_URI_CONFIG_KEY: "/tmp/mlruns"}
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "evaluate"
