import importlib
import logging
import os
import pathlib
import shutil
import sys
import urllib.parse

from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.pipelines.step import BaseStep
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

    def _run(self, output_directory):
        dataset_location = self.step_config["location"]
        dataset_format = self.step_config["format"]
        dataset_dst_path = os.path.abspath(os.path.join(output_directory, "dataset.parquet"))
        
        if dataset_format in [IngestStep._DATASET_FORMAT_SPARK_TABLE, IngestStep._DATASET_FORMAT_DELTA]:
           self._ingest_databricks_dataset(
                dataset_location=dataset_location,
                dataset_format=dataset_format,
                dataset_dst_path=dataset_dst_path,
            )
        else:
           self._ingest_dataset(
                dataset_location=dataset_location,
                dataset_format=dataset_format,
                dataset_dst_path=dataset_dst_path,
            )

        _logger.info("Resolved input data and stored it in '%s'", dataset_dst_path)

    def _ingest_dataset(self, dataset_location, dataset_format, dataset_dst_path):
        with TempDir(chdr=True) as tmpdir:
            local_dataset_path = download_artifacts(artifact_uri=dataset_location, dst_path=tmpdir.path())
            parquet_dataset_path = self._convert_dataset_to_parquet(
                local_dataset_path=local_dataset_path, dataset_format=dataset_format,
            )
            shutil.copyfile(parquet_dataset_path, dataset_dst_path)

    def _convert_dataset_to_parquet(self, local_dataset_path, dataset_format):
        import pandas as pd

        data_file_loader_method = self.step_config.get("loader_method")
        if data_file_loader_method:
            # TODO: Introduce a common utility for this
            sys.path.append(self.pipeline_root)
            data_file_loader_method_module, data_file_loader_method_name = data_file_loader_method.rsplit(".", 1)
            data_file_loader_method = getattr(importlib.import_module(data_file_loader_method_module), data_file_loader_method_name)

        if os.path.isdir(local_dataset_path):
            data_file_paths = list(pathlib.Path(local_dataset_path).glob(f"*.{dataset_format}"))
            if len(data_file_paths) == 0:
                raise MlflowException("TODO: SHOULDNT BE EMPTY")
        else:
            if not local_dataset_path.endswith(f".{dataset_format}"):
                raise MlflowException("TODO: WRONG FORMAT")
            data_file_paths = [local_dataset_path]

        aggregated_dataframe = None
        for data_file_path in data_file_paths:
            data_file_as_dataframe = self._load_data_file_as_pandas_dataframe(
                local_data_file_path=data_file_path,
                dataset_format=dataset_format,
                data_file_loader_method=data_file_loader_method,
            )
            aggregated_dataframe = (
                pd.concat([aggregated_dataframe, data_file_as_dataframe])
                if aggregated_dataframe is not None
                else data_file_as_dataframe
            )
           
        parquet_dataset_path = os.path.abspath("dataset.parquet")
        write_pandas_df_as_parquet(df=aggregated_dataframe, data_parquet_path=parquet_dataset_path)
        return parquet_dataset_path 

    def _load_data_file_as_pandas_dataframe(self, local_data_file_path, dataset_format, data_file_loader_method=None):
        if dataset_format == IngestStep._DATASET_FORMAT_PARQUET:
            return read_parquet_as_pandas_df(data_parquet_path=local_data_file_path)
        elif data_file_loader_method:
            return data_file_loader_method(local_data_file_path, dataset_format) 
        else:
            raise MlflowException("TODO: UNRECOGNIZED FORMAT")

    def _ingest_databricks_dataset(self, dataset_location, dataset_format, dataset_dst_path):
        raise NotImplemented 

    def inspect(self, output_directory):
        # Do step-specific code to inspect/materialize the output of the step
        _logger.info("ingest inspect code %s", output_directory)
        pass

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        step_config = {
            "location": IngestStep._sanitize_local_dataset_location_if_necessary(
                pipeline_root=pipeline_root,
                dataset_location=pipeline_config["data"]["location"],
            ),
            "format": pipeline_config["data"]["format"],
        }
        loader_method = pipeline_config["data"].get("loader_method")
        if loader_method:
            step_config["loader_method"] = loader_method
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
