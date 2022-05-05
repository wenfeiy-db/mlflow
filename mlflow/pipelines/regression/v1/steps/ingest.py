import importlib
import logging
import os
import pathlib
import sys

from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.cards import IngestCard
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, BAD_REQUEST
from mlflow.utils.file_utils import (
    TempDir,
    get_local_path_or_none,
    local_file_uri_to_path,
    write_pandas_df_as_parquet,
    read_parquet_as_pandas_df,
)

_logger = logging.getLogger(__name__)


class IngestStep(BaseStep):
    _DATASET_FORMAT_SPARK_TABLE = "spark_table"
    _DATASET_FORMAT_DELTA = "delta"
    _DATASET_FORMAT_PARQUET = "parquet"
    _DATASET_OUTPUT_NAME = "dataset.parquet"
    _STEP_CARD_OUTPUT_NAME = "card.html"

    def __init__(self, step_config, pipeline_root):
        super().__init__(step_config, pipeline_root)
        self.dataset_location = self.step_config["location"]
        self.dataset_format = self.step_config["format"]

    def _run(self, output_directory):
        dataset_dst_path = os.path.abspath(
            os.path.join(output_directory, IngestStep._DATASET_OUTPUT_NAME)
        )

        if self.dataset_format in [
            IngestStep._DATASET_FORMAT_SPARK_TABLE,
            IngestStep._DATASET_FORMAT_DELTA,
        ]:
            self._ingest_databricks_dataset(dataset_dst_path=dataset_dst_path)
        else:
            self._ingest_dataset(dataset_dst_path=dataset_dst_path)

        _logger.info("Successfully stored data in parquet format at '%s'", dataset_dst_path)

        step_card = IngestStep._build_step_card(
            dataset_src_location=self.dataset_location,
            ingested_parquet_dataset_path=dataset_dst_path,
        )
        with open(os.path.join(output_directory, IngestStep._STEP_CARD_OUTPUT_NAME), "w") as f:
            f.write(step_card.to_html())

    def _ingest_dataset(self, dataset_dst_path):
        with TempDir(chdr=True) as tmpdir:
            _logger.info("Resolving input data from '%s'", self.dataset_location)
            local_dataset_path = download_artifacts(
                artifact_uri=self.dataset_location, dst_path=tmpdir.path()
            )
            _logger.info("Resolved input data to '%s'", local_dataset_path)
            _logger.info("Converting dataset to parquet format, if necessary")
            self._convert_dataset_to_parquet(
                local_dataset_path=local_dataset_path,
                dataset_dst_path=dataset_dst_path,
            )

    def _convert_dataset_to_parquet(self, local_dataset_path, dataset_dst_path):
        import pandas as pd

        data_file_custom_loader_method = self.step_config.get("custom_loader_method")
        if data_file_custom_loader_method:
            # TODO: Introduce a common utility for this
            sys.path.append(self.pipeline_root)
            (
                data_file_custom_loader_method_module,
                data_file_custom_loader_method_name,
            ) = data_file_custom_loader_method.rsplit(".", 1)
            data_file_custom_loader_method = getattr(
                importlib.import_module(data_file_custom_loader_method_module),
                data_file_custom_loader_method_name,
            )

        if os.path.isdir(local_dataset_path):
            data_file_paths = list(
                pathlib.Path(local_dataset_path).glob(f"*.{self.dataset_format}")
            )
            if len(data_file_paths) == 0:
                raise MlflowException(
                    message=(
                        f"Did not find any data files with the specified format '{self.dataset_format}'"
                        f" in the resolved data directory with path '{local_dataset_path}'."
                        f" Directory contents: {os.listdir(local_dataset_path)}."
                    ),
                    error_code=INVALID_PARAMETER_VALUE,
                )
        else:
            if not local_dataset_path.endswith(f".{self.dataset_format}"):
                raise MlflowException(
                    message=(
                        f"Resolved data file with path '{local_dataset_path}' does not have the"
                        f" expected format '{self.dataset_format}'."
                    ),
                    error_code=INVALID_PARAMETER_VALUE,
                )
            data_file_paths = [local_dataset_path]

        aggregated_dataframe = None
        for data_file_path in data_file_paths:
            data_file_as_dataframe = IngestStep._load_data_file_as_pandas_dataframe(
                local_data_file_path=data_file_path,
                dataset_format=self.dataset_format,
                data_file_custom_loader_method=data_file_custom_loader_method,
            )
            aggregated_dataframe = (
                pd.concat([aggregated_dataframe, data_file_as_dataframe])
                if aggregated_dataframe is not None
                else data_file_as_dataframe
            )

        write_pandas_df_as_parquet(df=aggregated_dataframe, data_parquet_path=dataset_dst_path)

    @staticmethod
    def _load_data_file_as_pandas_dataframe(
        local_data_file_path, dataset_format, data_file_custom_loader_method=None
    ):
        if dataset_format == IngestStep._DATASET_FORMAT_PARQUET:
            try:
                return read_parquet_as_pandas_df(data_parquet_path=local_data_file_path)
            except Exception as e:
                raise MlflowException(
                    message=(
                        f"Failed to load data file at path '{local_data_file_path}' in Parquet"
                        f" format. Encountered exception: {e}"
                    ),
                    error_code=BAD_REQUEST,
                )
        elif data_file_custom_loader_method:
            try:
                return data_file_custom_loader_method(local_data_file_path, dataset_format)
            except NotImplementedError:
                raise MlflowException(
                    message=(
                        f"Unable to load data file at path '{local_data_file_path}' with format"
                        f" '{dataset_format}' using custom loader method"
                        f" '{data_file_custom_loader_method.__name__}' because it is not"
                        " supported. Please update the custom loader method to support this"
                        " format."
                    ),
                    error_code=INVALID_PARAMETER_VALUE,
                )
            except Exception:
                raise MlflowException(
                    message=(
                        f"Unable to load data file at path '{local_data_file_path}' with format"
                        f" '{dataset_format}' using custom loader method"
                        f" '{data_file_custom_loader_method.__name__}'. Encountered exception: {e}"
                    ),
                    error_code=BAD_REQUEST,
                )
        else:
            raise MlflowException(
                message=(
                    f"Unrecognized dataset format '{dataset_format}' for data file at path"
                    f" '{local_data_file_path}'. Please define and specify a `custom_loader_method`"
                    " that supports this format in 'pipeline.yaml'."
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )

    def _ingest_databricks_dataset(self, dataset_dst_path):
        raise NotImplementedError

    @staticmethod
    def _build_step_card(dataset_src_location, ingested_parquet_dataset_path):
        dataset_df = read_parquet_as_pandas_df(data_parquet_path=ingested_parquet_dataset_path)
        card = IngestCard()
        card.add_markdown(
            name="DATASET_SOURCE_LOCATION",
            markdown=f"**Dataset source location:** `{dataset_src_location}`",
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
        dataset_sample = dataset_df.sample(n=10, random_state=42).sort_index()
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
        step_config = {
            "location": IngestStep._sanitize_local_dataset_location_if_necessary(
                pipeline_root=pipeline_root,
                dataset_location=pipeline_config["data"]["location"],
            ),
            "format": pipeline_config["data"]["format"],
        }
        custom_loader_method = pipeline_config["data"].get("custom_loader_method")
        if custom_loader_method:
            step_config["custom_loader_method"] = custom_loader_method
        return cls(step_config, pipeline_root)

    @staticmethod
    def _sanitize_local_dataset_location_if_necessary(dataset_location, pipeline_root):
        local_dataset_path_or_uri_or_none = get_local_path_or_none(path_or_uri=dataset_location)
        if local_dataset_path_or_uri_or_none is None:
            return dataset_location

        # If the local dataset path is a file: URI, convert it to a filesystem path
        local_dataset_path = local_file_uri_to_path(uri=local_dataset_path_or_uri_or_none)
        local_dataset_path = pathlib.Path(local_dataset_path)
        if local_dataset_path.is_absolute():
            return str(local_dataset_path)
        else:
            # Use pathlib to join the local dataset relative path with the pipeline root
            # directory to correctly handle the case where the root path is Windows-formatted
            # and the local dataset relative path is POSIX-formatted
            return str(pathlib.Path(pipeline_root) / pathlib.Path(local_dataset_path))

    @property
    def name(self):
        return "ingest"
