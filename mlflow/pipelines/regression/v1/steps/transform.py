import importlib
import logging
import os
import sys
import time

import cloudpickle

from mlflow.exceptions import MlflowException, INVALID_PARAMETER_VALUE
from mlflow.pipelines.cards import BaseCard
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.pipelines.utils.tracking import get_pipeline_tracking_config

_logger = logging.getLogger(__name__)


class TransformStep(BaseStep):
    def __init__(self, step_config, pipeline_root):
        super().__init__(step_config, pipeline_root)
        self.run_end_time = None
        self.execution_duration = None
        self.target_col = self.step_config.get("target_col")
        (self.transformer_module_name, self.transformer_method_name,) = self.step_config[
            "transform_method"
        ].rsplit(".", 1)

    def _run(self, output_directory):
        import pandas as pd

        run_start_time = time.time()

        train_data_path = get_step_output_path(
            pipeline_name=self.hashed_pipeline_root,
            step_name="split",
            relative_path="train.parquet",
        )
        train_df = pd.read_parquet(train_data_path)

        validation_data_path = get_step_output_path(
            pipeline_name=self.hashed_pipeline_root,
            step_name="split",
            relative_path="validation.parquet",
        )
        validation_df = pd.read_parquet(validation_data_path)

        sys.path.append(self.pipeline_root)
        transformer_fn = getattr(
            importlib.import_module(self.transformer_module_name), self.transformer_method_name
        )
        transformer = transformer_fn()
        transformer.fit(train_df.drop(columns=[self.target_col]))

        def transform_dataset(dataset):
            features = dataset.drop(columns=[self.target_col])
            labels = dataset[self.target_col]
            transformed_feature_array = transformer.transform(features)
            num_features = transformed_feature_array.shape[1]
            # TODO: get the correct feature names from the transformer
            df = pd.DataFrame(
                transformed_feature_array, columns=[f"f_{i:03}" for i in range(num_features)]
            )
            df[self.target_col] = labels.values
            return df

        train_transformed = transform_dataset(train_df)
        validation_transformed = transform_dataset(validation_df)

        with open(os.path.join(output_directory, "transformer.pkl"), "wb") as f:
            cloudpickle.dump(transformer, f)

        train_transformed.to_parquet(
            os.path.join(output_directory, "transformed_training_data.parquet")
        )
        validation_transformed.to_parquet(
            os.path.join(output_directory, "transformed_validation_data.parquet")
        )

        self.run_end_time = time.time()
        self.execution_duration = self.run_end_time - run_start_time

        return self._build_profiles_and_card(
            train_df, train_transformed, validation_transformed, transformer
        )

    def _build_profiles_and_card(
        self, train_df, train_transformed, validation_transformed, transformer
    ) -> BaseCard:
        # Build card
        card = BaseCard(self.pipeline_name, self.name)

        # Tab 1 and 2: build profiles for train_transformed, validation_transformed
        from pandas_profiling import ProfileReport

        train_transformed_profile = ProfileReport(
            train_transformed, title="Profile of Train Transformed Dataset", minimal=True
        )
        validation_transformed_profile = ProfileReport(
            validation_transformed, title="Profile of Validation Transformed Dataset", minimal=True
        )
        card.add_tab("Data Profile (Train Transformed)", "{{PROFILE}}").add_pandas_profile(
            "PROFILE", train_transformed_profile
        )
        card.add_tab("Data Profile (Validation Transformed)", "{{PROFILE}}").add_pandas_profile(
            "PROFILE", validation_transformed_profile
        )

        # Tab 3: transformer diagram
        from sklearn.utils import estimator_html_repr
        from sklearn import set_config

        set_config(display="diagram")
        transformer_repr = estimator_html_repr(transformer)
        card.add_tab("Transformer", "{{TRANSFORMER}}").add_html("TRANSFORMER", transformer_repr)

        # Tab 4: transformer output schema
        from sklearn.pipeline import make_pipeline

        # Construct a fake pipeline containing the transformer and a passthrough estimator.
        try:
            p = make_pipeline([transformer, "passthrough"])
            output_schema = list(p[:-1].get_feature_names_out(train_df.columns))
            card.add_tab("Output Schema", "{{OUTPUT_SCHEMA}}").add_html(
                "OUTPUT_SCHEMA", output_schema
            )
        except Exception as e:
            card.add_tab("Output Schema", "{{OUTPUT_SCHEMA}}").add_html(
                "OUTPUT_SCHEMA", f"Failed to extract transformer schema. Error: {e}"
            )

        # Tab 5: run summary
        (
            card.add_tab(
                "Run Summary",
                """
                {{ RAW_FEATURES_IN_TRANSFORMER }}
                {{ EXE_DURATION }}
                {{ LAST_UPDATE_TIME }}
                """,
            ).add_markdown(
                "RAW_FEATURES_IN_TRANSFORMER",
                f"**Feature set before transformation:** `{train_df.columns}`",
            )
        )

        return card

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        try:
            step_config = pipeline_config["steps"]["transform"]
            step_config.update(
                get_pipeline_tracking_config(
                    pipeline_root_path=pipeline_root,
                    pipeline_config=pipeline_config,
                ).to_dict()
            )
        except KeyError:
            raise MlflowException(
                "Config for transform step is not found.", error_code=INVALID_PARAMETER_VALUE
            )
        step_config["target_col"] = pipeline_config.get("target_col")
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "transform"
