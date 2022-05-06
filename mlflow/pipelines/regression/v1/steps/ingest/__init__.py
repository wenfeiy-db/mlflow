import logging
import os
from abc import abstractmethod

from mlflow.exceptions import MlflowException
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.cards import IngestCard
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, INTERNAL_ERROR
from mlflow.utils.file_utils import read_parquet_as_pandas_df
from mlflow.pipelines.regression.v1.steps.ingest.datasets import (
    ParquetDataset,
    DeltaTableDataset,
    SparkSqlDataset,
    CustomDataset,
)

_logger = logging.getLogger(__name__)


class IngestStep(BaseStep):
    _DATASET_FORMAT_SPARK_TABLE = "spark_table"
    _DATASET_FORMAT_DELTA = "delta"
    _DATASET_FORMAT_PARQUET = "parquet"
    _DATASET_OUTPUT_NAME = "dataset.parquet"
    _STEP_CARD_OUTPUT_NAME = "card.html"
    _SUPPORTED_DATASETS = [
        ParquetDataset,
        DeltaTableDataset,
        SparkSqlDataset,
        # NB: The custom dataset is deliberately listed last as a catch-all for any
        # format not matched by the datasets above. When mapping a format to a dataset,
        # datasets are explored in the listed order
        CustomDataset,
    ]

    def __init__(self, step_config, pipeline_root):
        super().__init__(step_config, pipeline_root)

        dataset_format = step_config.get("format")
        if not dataset_format:
            raise MlflowException(
                message="TODO: NEED FORMAT",
                error_code=INVALID_PARAMETER_VALUE,
            )

        for dataset_class in IngestStep._SUPPORTED_DATASETS:
            if dataset_class.matches_format(dataset_format):
                self.dataset = dataset_class.from_config(
                    config=step_config,
                    pipeline_root=pipeline_root,
                )
                break
        else:
            raise MlflowException(
                "TODO: UNRECOGNIZED FORMAT CATCH-ALL",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def _run(self, output_directory):
        dataset_dst_path = os.path.abspath(
            os.path.join(output_directory, IngestStep._DATASET_OUTPUT_NAME)
        )
        self.dataset.resolve_to_parquet(dst_path=dataset_dst_path)
        _logger.info("Successfully stored data in parquet format at '%s'", dataset_dst_path)

        step_card = IngestStep._build_step_card(
            ingested_parquet_dataset_path=dataset_dst_path,
            dataset_src_location=getattr(self.dataset, "location", None),
            dataset_sql=getattr(self.dataset, "sql", None),
        )
        with open(os.path.join(output_directory, IngestStep._STEP_CARD_OUTPUT_NAME), "w") as f:
            f.write(step_card.to_html())

    @staticmethod
    def _build_step_card(
        ingested_parquet_dataset_path, dataset_src_location=None, dataset_sql=None
    ):
        if dataset_src_location is None and dataset_sql is None:
            raise MlflowException("TODO: MUST SPECIFY ONE", error_code=INTERNAL_ERROR)

        dataset_df = read_parquet_as_pandas_df(data_parquet_path=ingested_parquet_dataset_path)
        card = IngestCard()
        card.add_markdown(
            name="DATASET_SOURCE",
            markdown=(
                f"**Dataset source location:** `{dataset_src_location}`"
                if dataset_src_location is not None
                else f"**Dataset SQL:** `{dataset_sql}`"
            ),
        )
        card.add_markdown(
            name="RESOLVED_DATASET_LOCATION",
            markdown=f"**Ingested dataset path:** `{ingested_parquet_dataset_path}`",
        )
        card.add_markdown(
            name="DATASET_NUM_ROWS",
            markdown=f"**Number of rows:** {len(dataset_df)}",
        )
        dataset_size = IngestStep._get_dataset_size(dataset_path=ingested_parquet_dataset_path)
        card.add_markdown(
            name="DATASET_SIZE",
            markdown=f"**Size:** {dataset_size}",
        )
        dataset_types = dataset_df.dtypes.to_frame(name="column type").transpose()
        dataset_types = dataset_types.reset_index(level=0)
        dataset_types = dataset_types.rename(columns={"index": "column name"})
        dataset_types_styler = (
            dataset_types.style.set_properties(**{"text-align": "center"})
            .hide(axis="index")
            .set_table_styles(
                [
                    {"selector": "", "props": [("border", "1px solid grey")]},
                    {"selector": "tbody td", "props": [("border", "1px solid grey")]},
                    {"selector": "th", "props": [("border", "1px solid grey")]},
                ]
            )
        )
        card.add_artifact(
            name="DATASET_SCHEMA",
            artifact=dataset_types_styler.to_html(),
            artifact_format="html",
        )
        dataset_sample = dataset_df.sample(n=min(len(dataset_df), 10), random_state=42).sort_index()
        dataset_sample_styler = (
            dataset_sample.style.set_properties(**{"text-align": "center"})
            .format(precision=2)
            .set_table_styles(
                [
                    {"selector": "", "props": [("border", "1px solid grey")]},
                    {"selector": "tbody td", "props": [("border", "1px solid grey")]},
                    {"selector": "th", "props": [("border", "1px solid grey")]},
                ]
            )
        )
        card.add_artifact(
            name="DATASET_SAMPLE",
            artifact=dataset_sample_styler.to_html(),
            artifact_format="html",
        )
        return card

    @staticmethod
    def _get_dataset_size(dataset_path):
        kb = 10**3
        mb = 10**6
        gb = 10**9

        size = os.path.getsize(dataset_path)
        if size >= gb:
            return "{:0.2f} GB".format(size / gb)
        elif size >= mb:
            return "{:0.2f} MB".format(size / mb)
        elif size >= kb:
            return "{:0.2f} KB".format(size / kb)
        else:
            return f"{size} B"

    def inspect(self, output_directory):
        parquet_dataset_path = os.path.join(output_directory, IngestStep._DATASET_OUTPUT_NAME)
        return IngestStep._build_step_card(
            dataset_src_location=self.dataset_location,
            ingested_parquet_dataset_path=parquet_dataset_path,
        )

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        if "data" not in pipeline_config:
            raise MlflowException(
                message="TODO: NEED DATA",
                error_code=INVALID_PARAMETER_VALUE,
            )

        return cls(
            step_config=pipeline_config["data"],
            pipeline_root=pipeline_root,
        )

    @property
    def name(self):
        return "ingest"
