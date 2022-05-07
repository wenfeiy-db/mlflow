import importlib
import logging
import os
import pathlib
import sys
from abc import abstractmethod

from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
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
            dataset_location=location,
            pipeline_root=pipeline_root,
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
            return str(pathlib.Path(pipeline_root) / local_dataset_path)

    @staticmethod
    @abstractmethod
    def matches_format(dataset_format):
        pass


class _PandasParseableDataset(_LocationBasedDataset):
    def resolve_to_parquet(self, dst_path):
        import pandas as pd

        with TempDir(chdr=True) as tmpdir:
            _logger.info("Resolving input data from '%s'", self.location)
            local_dataset_path = download_artifacts(
                artifact_uri=self.location, dst_path=tmpdir.path()
            )

            if os.path.isdir(local_dataset_path):
                # NB: Sort the file names alphanumerically to ensure a consistent
                # ordering across invocations
                data_file_paths = sorted(
                    list(pathlib.Path(local_dataset_path).glob(f"*.{self.dataset_format}"))
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


class ParquetDataset(_PandasParseableDataset):
    def _load_file_as_pandas_dataframe(self, local_data_file_path):
        return read_parquet_as_pandas_df(data_parquet_path=local_data_file_path)

    def matches_format(dataset_format):
        return dataset_format == "parquet"


class CustomDataset(_PandasParseableDataset):
    def __init__(self, location, dataset_format, custom_loader_method, pipeline_root):
        super().__init__(
            location=location, dataset_format=dataset_format, pipeline_root=pipeline_root
        )
        self.pipeline_root = pipeline_root
        (
            self.custom_loader_module_name,
            self.custom_loader_method_name,
        ) = custom_loader_method.rsplit(".", 1)

    def _load_file_as_pandas_dataframe(self, local_data_file_path):
        try:
            sys.path.append(self.pipeline_root)
            custom_loader_method = getattr(
                importlib.import_module(self.custom_loader_module_name),
                self.custom_loader_method_name,
            )
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
        except Exception as e:
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
            custom_loader_method=cls._get_required_config(
                config=config, key="custom_loader_method"
            ),
            pipeline_root=pipeline_root,
        )

    @staticmethod
    def matches_format(dataset_format):
        return dataset_format is not None


class _SparkDatasetMixin:
    """
    TODO: DOCS: MUST BE MIXED INTO A SUBCLASS OF `_DATASET`
    """

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


class DeltaTableDataset(_SparkDatasetMixin, _LocationBasedDataset):
    def resolve_to_parquet(self, dst_path):
        spark_session = self._get_spark_session()
        spark_df = spark_session.read.format("delta").load(self.location)
        if len(spark_df.columns) > 0:
            # Sort across columns in hopes of achieving a consistent ordering at ingest
            spark_df = spark_df.orderBy(spark_df.columns)
        spark_df.write.parquet(dst_path)

    @staticmethod
    def matches_format(dataset_format):
        return dataset_format == "delta"


class SparkSqlDataset(_SparkDatasetMixin, _Dataset):
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
