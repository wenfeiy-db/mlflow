import os
import pytest
import shutil

import pandas as pd
from pyspark.sql import SparkSession

from mlflow.pipelines.regression.v1.steps.ingest import IngestStep
from mlflow.utils.file_utils import TempDir

from tests.pipelines.helper_functions import (
    enter_pipeline_example_directory,
)


@pytest.fixture
def enter_ingest_test_pipeline_directory(enter_pipeline_example_directory):
    pipeline_example_root_path = enter_pipeline_example_directory

    with TempDir(chdr=True) as tmp:
        ingest_test_pipeline_path = tmp.path("test_ingest_pipeline")
        shutil.copytree(pipeline_example_root_path, ingest_test_pipeline_path)
        os.chdir(ingest_test_pipeline_path)
        yield os.getcwd()


@pytest.fixture
def pandas_df():
    df = pd.DataFrame.from_dict(
        {
            "A": ["x", "y", "z"],
            "B": [1, 2, 3],
            "C": [-9.2, 82.5, 3.40],
        }
    )
    df.index.rename("index", inplace=True)
    return df


@pytest.fixture(scope="module", autouse=True)
def spark_session():
    session = (
        SparkSession.builder.master("local[*]")
        .config("spark.jars.packages", "io.delta:delta-core_2.12:1.2.1")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture()
def spark_df(spark_session):
    return spark_session.createDataFrame(
        [
            (0, "a b c d e spark", 1.0),
            (1, "b d", 0.0),
            (2, "spark f g h", 1.0),
            (3, "hadoop mapreduce", 0.0),
        ],
        ["id", "text", "label"],
    ).cache()


@pytest.mark.parametrize("use_relative_path", [False, True])
@pytest.mark.parametrize("multiple_files", [False, True])
def test_ingests_parquet_successfully(
    enter_ingest_test_pipeline_directory, use_relative_path, multiple_files, pandas_df, tmp_path
):
    ingest_test_pipeline_root_path = enter_ingest_test_pipeline_directory
    if multiple_files:
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()
        pandas_df_part1 = pandas_df[:1]
        pandas_df_part2 = pandas_df[1:]
        pandas_df_part1.to_parquet(dataset_path / "df1.parquet")
        pandas_df_part2.to_parquet(dataset_path / "df2.parquet")
    else:
        dataset_path = tmp_path / "df.parquet"
        pandas_df.to_parquet(dataset_path)

    if use_relative_path:
        dataset_path = os.path.relpath(dataset_path)

    IngestStep.from_pipeline_config(
        pipeline_config={
            "data": {
                "format": "parquet",
                "location": str(dataset_path),
            }
        },
        pipeline_root=ingest_test_pipeline_root_path,
    ).run(output_directory=tmp_path)

    reloaded_df = pd.read_parquet(str(tmp_path / "dataset.parquet"))
    pd.testing.assert_frame_equal(reloaded_df, pandas_df)


@pytest.mark.parametrize("use_relative_path", [False, True])
@pytest.mark.parametrize("multiple_files", [False, True])
def test_ingests_csv_successfully(
    enter_ingest_test_pipeline_directory, use_relative_path, multiple_files, pandas_df, tmp_path
):
    ingest_test_pipeline_root_path = enter_ingest_test_pipeline_directory
    if multiple_files:
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()
        pandas_df_part1 = pandas_df[:1]
        pandas_df_part2 = pandas_df[1:]
        pandas_df_part1.to_csv(dataset_path / "df1.csv")
        pandas_df_part2.to_csv(dataset_path / "df2.csv")
    else:
        dataset_path = tmp_path / "df.csv"
        pandas_df.to_csv(dataset_path)

    if use_relative_path:
        dataset_path = os.path.relpath(dataset_path)

    IngestStep.from_pipeline_config(
        pipeline_config={
            "data": {
                "format": "csv",
                "location": str(dataset_path),
                "custom_loader_method": "steps.ingest.load_file_as_dataframe",
            }
        },
        pipeline_root=ingest_test_pipeline_root_path,
    ).run(output_directory=tmp_path)

    reloaded_df = pd.read_parquet(str(tmp_path / "dataset.parquet"))
    pd.testing.assert_frame_equal(reloaded_df, pandas_df)


def custom_load_file_as_dataframe(file_path, file_format):  # pylint: disable=unused-argument
    return pd.read_csv(file_path, sep="#", index_col=0)


@pytest.mark.parametrize("use_relative_path", [False, True])
@pytest.mark.parametrize("multiple_files", [False, True])
def test_ingests_custom_format_successfully(
    enter_ingest_test_pipeline_directory, use_relative_path, multiple_files, pandas_df, tmp_path
):
    ingest_test_pipeline_root_path = enter_ingest_test_pipeline_directory
    if multiple_files:
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()
        pandas_df_part1 = pandas_df[:1]
        pandas_df_part2 = pandas_df[1:]
        pandas_df_part1.to_csv(dataset_path / "df1.fooformat", sep="#")
        pandas_df_part2.to_csv(dataset_path / "df2.fooformat", sep="#")
    else:
        dataset_path = tmp_path / "df.fooformat"
        pandas_df.to_csv(dataset_path, sep="#")

    if use_relative_path:
        dataset_path = os.path.relpath(dataset_path)

    IngestStep.from_pipeline_config(
        pipeline_config={
            "data": {
                "format": "fooformat",
                "location": str(dataset_path),
                "custom_loader_method": "tests.pipelines.test_ingest_step.custom_load_file_as_dataframe",
            }
        },
        pipeline_root=ingest_test_pipeline_root_path,
    ).run(output_directory=tmp_path)

    reloaded_df = pd.read_parquet(str(tmp_path / "dataset.parquet"))
    pd.testing.assert_frame_equal(reloaded_df, pandas_df)


def test_ingests_spark_sql_successfully(enter_ingest_test_pipeline_directory, spark_df, tmp_path):
    ingest_test_pipeline_root_path = enter_ingest_test_pipeline_directory
    spark_df.write.saveAsTable("test_table")

    IngestStep.from_pipeline_config(
        pipeline_config={
            "data": {
                "format": "spark_sql",
                "sql": "SELECT * FROM test_table ORDER BY id",
            }
        },
        pipeline_root=ingest_test_pipeline_root_path,
    ).run(output_directory=tmp_path)

    reloaded_df = pd.read_parquet(str(tmp_path / "dataset.parquet"))
    pd.testing.assert_frame_equal(reloaded_df, spark_df.toPandas())


@pytest.mark.parametrize("use_relative_path", [False, True])
def test_ingests_delta_successfully(
    enter_ingest_test_pipeline_directory, use_relative_path, spark_df, tmp_path
):
    ingest_test_pipeline_root_path = enter_ingest_test_pipeline_directory
    dataset_path = tmp_path / "test.delta"
    spark_df.write.format("delta").save(str(dataset_path))
    if use_relative_path:
        dataset_path = os.path.relpath(dataset_path)

    IngestStep.from_pipeline_config(
        pipeline_config={
            "data": {
                "format": "delta",
                "location": str(dataset_path),
            }
        },
        pipeline_root=ingest_test_pipeline_root_path,
    ).run(output_directory=tmp_path)

    reloaded_df = pd.read_parquet(str(tmp_path / "dataset.parquet"))
    pd.testing.assert_frame_equal(reloaded_df, spark_df.toPandas())
