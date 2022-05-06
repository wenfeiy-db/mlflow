import importlib
import logging
import os
import pathlib
import sys
from abc import abstractmethod

from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.cards import IngestCard
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, BAD_REQUEST, INTERNAL_ERROR
from mlflow.utils.file_utils import (
    TempDir,
    get_local_path_or_none,
    local_file_uri_to_path,
    write_pandas_df_as_parquet,
    read_parquet_as_pandas_df,
)
from mlflow.utils._spark_utils import _get_active_spark_session

_logger = logging.getLogger(__name__)


class _Dataset:

    def __init__(self, dataset_format):
        self.dataset_format = dataset_format

    @abstractmethod
    def resolve_to_parquet(self, dst_path):
        pass

    @classmethod
    def from_config(cls, config, pipeline_root):
        if not cls.matches_format(config.get("format")):
            raise MlflowException(
                f"Invalid format {config.get('format')} for dataset {cls}",
                error_code=INTERNAL_ERROR,
            )
        return cls._from_config(config, pipeline_root)

    @classmethod
    @abstractmethod
    def _from_config(cls, config, pipeline_root):
        pass

    @staticmethod
    @abstractmethod
    def matches_format(dataset_format):
        pass


    @classmethod
    def _get_required_config(cls, config, key):
        try:
            return config[key]
        except KeyError:
            raise MlflowException(
                f"The `{key}` configuration key must be specified for dataset with"
                f" format '{config.get('format')}'"
            ) from None


class _LocationBasedDataset(_Dataset):

    def __init__(self, location, dataset_format, pipeline_root):
        super().__init__(dataset_format=dataset_format)
        self.location = _LocationBasedDataset._sanitize_local_dataset_location_if_necessary(
            dataset_location=location, pipeline_root=pipeline_root,
        )

    @abstractmethod
    def resolve_to_parquet(self, dst_path):
        pass

    @classmethod
    def _from_config(cls, config, pipeline_root):
        return cls(
            location=cls._get_required_config(config=config, key="location"),
            pipeline_root=pipeline_root,
            dataset_format=cls._get_required_config(config=config, key="format"),
        )

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

    @staticmethod
    @abstractmethod
    def matches_format(dataset_format):
        pass


class _PandasParseableDataset(_LocationBasedDataset):

    def resolve_to_parquet(self, dst_path):
        with TempDir(chdr=True) as tmpdir:
            _logger.info("Resolving input data from '%s'", self.location)
            local_dataset_path = download_artifacts(
                artifact_uri=self.location, dst_path=tmpdir.path()
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

            _logger.info("Resolved input data to '%s'", local_dataset_path)
            _logger.info("Converting dataset to parquet format, if necessary")
            aggregated_dataframe = None
            for data_file_path in data_file_paths:
                data_file_as_dataframe = self._load_file_as_pandas_dataframe(
                    local_data_file_path=data_file_path,
                )
                aggregated_dataframe = (
                    pd.concat([aggregated_dataframe, data_file_as_dataframe])
                    if aggregated_dataframe is not None
                    else data_file_as_dataframe
                )

            write_pandas_df_as_parquet(df=aggregated_dataframe, data_parquet_path=dst_path)

    @abstractmethod
    def _load_file_as_pandas_dataframe(self, local_data_file_path):
        pass

    @staticmethod
    @abstractmethod
    def matches_format(dataset_format):
        pass


class _ParquetDataset(_PandasParseableDataset):
    def _load_file_as_pandas_dataframe(self, local_data_file_path):
        return read_parquet_as_pandas_df(data_parquet_path=local_data_file_path)

    def matches_format(dataset_format):
        return dataset_format == "parquet"


class _CustomDataset(_PandasParseableDataset):
    def __init__(self, location, dataset_format, custom_loader_method, pipeline_root):
        super().__init__(location=location, dataset_format=dataset_format, pipeline_root=pipeline_root)
        self.pipeline_root = pipeline_root
        self.custom_loader_module_name, self.custom_loader_method_name = custom_loader_method.rsplit(".", 1)

    def _load_file_as_pandas_dataframe(self, local_data_file_path):
        try:
            sys.path.append(self.pipeline_root)
            custom_loader_method = getattr(importlib.import_module(self.custom_loader_module_name), self.custom_loader_method_name)
        except Exception:
            raise MlflowException(
                message="TODO: FAILED TO LOAD LOADER FUNCTION....",
                error_code=BAD_REQUEST,
            )

        try:
            return custom_loader_method(local_data_file_path, self.dataset_format)
        except NotImplementedError:
            raise MlflowException(
                message=(
                    f"Unable to load data file at path '{local_data_file_path}' with format"
                    f" '{self.dataset_format}' using custom loader method"
                    f" '{custom_loader_method.__name__}' because it is not"
                    " supported. Please update the custom loader method to support this"
                    " format."
                ),
                error_code=INVALID_PARAMETER_VALUE,
            ) from None
        except Exception:
            raise MlflowException(
                message=(
                    f"Unable to load data file at path '{local_data_file_path}' with format"
                    f" '{self.dataset_format}' using custom loader method"
                    f" '{custom_loader_method.__name__}'. Encountered exception: {e}"
                ),
                error_code=BAD_REQUEST,
            ) from None

    @classmethod
    def _from_config(cls, config, pipeline_root):
        return cls(
            location=cls._get_required_config(config=config, key="location"),
            dataset_format=cls._get_required_config(config=config, key="format"),
            custom_loader_method=cls._get_required_config(config=config, key="custom_loader_method"),
            pipeline_root=pipeline_root,
        )

    @staticmethod
    def matches_format(dataset_format):
        return dataset_format is not None


class _SparkDatasetMixin:
    """
    TODO: DOCS: MUST BE MIXED INTO A SUBCLASS OF `_DATASET`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_spark_session(self):
        try:
            return _get_active_spark_session()
        except Exception as e:
            raise MlflowException(
                message=(
                    f"Encountered an error while searching for an active Spark session to"
                    f" load the dataset with format '{self.dataset_format}'. Please create a"
                    f" Spark session and try again."
                ),
                error_code=BAD_REQUEST,
            ) from e


class _DeltaTableDataset(_SparkDatasetMixin, _LocationBasedDataset):
    def resolve_to_parquet(self, dst_path):
        spark_session = self._get_spark_session()
        spark_df = spark_session.read.format("delta").load(self.location)
        spark_df.write.parquet(dst_path)

    @staticmethod
    def matches_format(dataset_format):
        return dataset_format == "delta"


class _SparkSqlDataset(_SparkDatasetMixin, _Dataset):
    def __init__(self, sql, dataset_format):
        super().__init__(dataset_format=dataset_format)
        self.sql = sql

    def resolve_to_parquet(self, dst_path):
        spark_session = self._get_spark_session()
        spark_df = spark_session.sql(self.sql)
        spark_df.write.parquet(dst_path)

    @classmethod
    def _from_config(cls, config, pipeline_root):
        return cls(
            sql=cls._get_required_config(config=config, key="sql"),
            dataset_format=cls._get_required_config(config=config, key="format"),
        )

    @staticmethod
    def matches_format(dataset_format):
        return dataset_format == "spark_sql"


class IngestStep(BaseStep):
    _DATASET_FORMAT_SPARK_TABLE = "spark_table"
    _DATASET_FORMAT_DELTA = "delta"
    _DATASET_FORMAT_PARQUET = "parquet"
    _DATASET_OUTPUT_NAME = "dataset.parquet"
    _STEP_CARD_OUTPUT_NAME = "card.html"
    _SUPPORTED_DATASETS = [
        _ParquetDataset,
        _DeltaTableDataset,
        _SparkSqlDataset,
        # NB: The custom dataset is deliberately listed last as a catch-all for any
        # format not matched by the datasets above. When mapping a format to a dataset,
        # datasets are explored in the listed order
        _CustomDataset,
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
    def _build_step_card(ingested_parquet_dataset_path, dataset_src_location=None, dataset_sql=None):
        if dataset_src_location is None and dataset_sql is None:
            raise MlflowException("TODO: MUST SPECIFY ONE", error_code=INTERNAL_ERROR)

        dataset_df = read_parquet_as_pandas_df(data_parquet_path=ingested_parquet_dataset_path)
        card = IngestCard()
        card.add_markdown(
            name="DATASET_SOURCE",
            markdown=(
                f"**Dataset source location:** `{dataset_src_location}`"
                if dataset_src_location is not None else
                f"**Dataset SQL:** `{dataset_sql}`"
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
