import os
import shutil
from unittest import mock

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from mlflow.exceptions import MlflowException
from mlflow.pipelines.regression.v1.cards.ingest import IngestCard
from mlflow.pipelines.regression.v1.steps.ingest import IngestStep
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.utils.file_utils import TempDir

from tests.helper_functions import mock_s3_bucket  # pylint: disable=unused-import

# pylint: disable=unused-import
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
@pytest.mark.usefixtures("enter_ingest_test_pipeline_directory")
def test_ingests_parquet_successfully(use_relative_path, multiple_files, pandas_df, tmp_path):
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
        pipeline_root=os.getcwd(),
    ).run(output_directory=tmp_path)

    reloaded_df = pd.read_parquet(str(tmp_path / "dataset.parquet"))
    pd.testing.assert_frame_equal(reloaded_df, pandas_df)


@pytest.mark.parametrize("use_relative_path", [False, True])
@pytest.mark.parametrize("multiple_files", [False, True])
@pytest.mark.usefixtures("enter_ingest_test_pipeline_directory")
def test_ingests_csv_successfully(use_relative_path, multiple_files, pandas_df, tmp_path):
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
        pipeline_root=os.getcwd(),
    ).run(output_directory=tmp_path)

    reloaded_df = pd.read_parquet(str(tmp_path / "dataset.parquet"))
    pd.testing.assert_frame_equal(reloaded_df, pandas_df)


def custom_load_file_as_dataframe(file_path, file_format):  # pylint: disable=unused-argument
    return pd.read_csv(file_path, sep="#", index_col=0)


@pytest.mark.parametrize("use_relative_path", [False, True])
@pytest.mark.parametrize("multiple_files", [False, True])
@pytest.mark.usefixtures("enter_ingest_test_pipeline_directory")
def test_ingests_custom_format_successfully(use_relative_path, multiple_files, pandas_df, tmp_path):
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
                "custom_loader_method": (
                    "tests.pipelines.test_ingest_step.custom_load_file_as_dataframe"
                ),
            }
        },
        pipeline_root=os.getcwd(),
    ).run(output_directory=tmp_path)

    reloaded_df = pd.read_parquet(str(tmp_path / "dataset.parquet"))
    pd.testing.assert_frame_equal(reloaded_df, pandas_df)


@pytest.mark.usefixtures("enter_ingest_test_pipeline_directory")
def test_ingest_throws_for_custom_dataset_when_custom_loader_function_cannot_be_imported(
    pandas_df, tmp_path
):
    dataset_path = tmp_path / "df.fooformat"
    pandas_df.to_csv(dataset_path, sep="#")

    with pytest.raises(MlflowException, match="Failed to import custom dataset loader function"):
        IngestStep.from_pipeline_config(
            pipeline_config={
                "data": {
                    "format": "fooformat",
                    "location": str(dataset_path),
                    "custom_loader_method": "non.existent.module.non.existent.method",
                }
            },
            pipeline_root=os.getcwd(),
        ).run(output_directory=tmp_path)


@pytest.mark.usefixtures("enter_ingest_test_pipeline_directory")
def test_ingest_throws_for_custom_dataset_when_custom_loader_function_not_implemented_for_format(
    pandas_df, tmp_path
):
    dataset_path = tmp_path / "df.fooformat"
    pandas_df.to_csv(dataset_path, sep="#")

    with pytest.raises(
        MlflowException, match="Please update the custom loader method to support this format"
    ):
        IngestStep.from_pipeline_config(
            pipeline_config={
                "data": {
                    "format": "fooformat",
                    "location": str(dataset_path),
                    "custom_loader_method": "steps.ingest.load_file_as_dataframe",
                }
            },
            pipeline_root=os.getcwd(),
        ).run(output_directory=tmp_path)


@pytest.mark.usefixtures("enter_ingest_test_pipeline_directory")
def test_ingest_throws_for_custom_dataset_when_custom_loader_function_throws_unexpectedly(
    pandas_df, tmp_path
):
    dataset_path = tmp_path / "df.fooformat"
    pandas_df.to_csv(dataset_path, sep="#")

    with mock.patch(
        "tests.pipelines.test_ingest_step.custom_load_file_as_dataframe",
        side_effect=Exception("Failed to load!"),
    ) as mock_custom_loader, pytest.raises(
        MlflowException, match="Unable to load data file at path.*using custom loader method"
    ):
        setattr(mock_custom_loader, "__name__", "custom_load_file_as_dataframe")
        IngestStep.from_pipeline_config(
            pipeline_config={
                "data": {
                    "format": "fooformat",
                    "location": str(dataset_path),
                    "custom_loader_method": (
                        "tests.pipelines.test_ingest_step.custom_load_file_as_dataframe"
                    ),
                }
            },
            pipeline_root=os.getcwd(),
        ).run(output_directory=tmp_path)


@pytest.mark.usefixtures("enter_ingest_test_pipeline_directory")
def test_ingests_remote_s3_datasets_successfully(mock_s3_bucket, pandas_df, tmp_path):
    with mock.patch("shutil.which") as patched_which:
        patched_which.return_value(None)
        dataset_path = tmp_path / "df.parquet"
        pandas_df.to_parquet(dataset_path)
        S3ArtifactRepository(f"s3://{mock_s3_bucket}").log_artifact(str(dataset_path))

        IngestStep.from_pipeline_config(
            pipeline_config={
                "data": {
                    "format": "parquet",
                    "location": f"s3://{mock_s3_bucket}/df.parquet",
                }
            },
            pipeline_root=os.getcwd(),
        ).run(output_directory=tmp_path)

        reloaded_df = pd.read_parquet(str(tmp_path / "dataset.parquet"))
        pd.testing.assert_frame_equal(reloaded_df, pandas_df)


@pytest.mark.usefixtures("enter_ingest_test_pipeline_directory")
def test_ingests_remote_http_datasets_successfully(tmp_path):
    dataset_url = "https://raw.githubusercontent.com/mlflow/mlflow/594a08f2a49c5754bb65d76cd719c15c5b8266e9/examples/sklearn_elasticnet_wine/wine-quality.csv"
    IngestStep.from_pipeline_config(
        pipeline_config={
            "data": {
                "format": "csv",
                "location": dataset_url,
                "custom_loader_method": "steps.ingest.load_file_as_dataframe",
            }
        },
        pipeline_root=os.getcwd(),
    ).run(output_directory=tmp_path)

    reloaded_df = pd.read_parquet(str(tmp_path / "dataset.parquet"))
    pd.testing.assert_frame_equal(reloaded_df, pd.read_csv(dataset_url, index_col=0))


@pytest.mark.usefixtures("enter_ingest_test_pipeline_directory")
def test_ingests_spark_sql_successfully(spark_df, tmp_path):
    spark_df.write.saveAsTable("test_table")

    IngestStep.from_pipeline_config(
        pipeline_config={
            "data": {
                "format": "spark_sql",
                "sql": "SELECT * FROM test_table ORDER BY id",
            }
        },
        pipeline_root=os.getcwd(),
    ).run(output_directory=tmp_path)

    # Spark DataFrames are not ingested with a consistent row order, as doing so would incur a
    # substantial performance cost. Accordingly, we sort the ingested DataFrame and the original
    # DataFrame on the `id` column and reset the DataFrame index to achieve a consistent ordering
    # before testing their equivalence
    reloaded_df = (
        pd.read_parquet(str(tmp_path / "dataset.parquet"))
        .sort_values(by="id")
        .reset_index(drop=True)
    )
    spark_to_pandas_df = spark_df.toPandas().sort_values(by="id").reset_index(drop=True)
    pd.testing.assert_frame_equal(reloaded_df, spark_to_pandas_df)


@pytest.mark.parametrize("use_relative_path", [False, True])
@pytest.mark.usefixtures("enter_ingest_test_pipeline_directory")
def test_ingests_delta_successfully(use_relative_path, spark_df, tmp_path):
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
        pipeline_root=os.getcwd(),
    ).run(output_directory=tmp_path)

    # Spark DataFrames are not ingested with a consistent row order, as doing so would incur a
    # substantial performance cost. Accordingly, we sort the ingested DataFrame and the original
    # DataFrame on the `id` column and reset the DataFrame index to achieve a consistent ordering
    # before testing their equivalence
    reloaded_df = (
        pd.read_parquet(str(tmp_path / "dataset.parquet"))
        .sort_values(by="id")
        .reset_index(drop=True)
    )
    spark_to_pandas_df = spark_df.toPandas().sort_values(by="id").reset_index(drop=True)
    pd.testing.assert_frame_equal(reloaded_df, spark_to_pandas_df)


@pytest.mark.usefixtures("enter_ingest_test_pipeline_directory")
def test_ingest_directory_ignores_files_that_do_not_match_dataset_format(pandas_df, tmp_path):
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    pandas_df_part1 = pandas_df[:1]
    pandas_df_part2 = pandas_df[1:]
    pandas_df_part1.to_parquet(dataset_path / "df1.parquet")
    pandas_df_part2.to_parquet(dataset_path / "df2.parquet")
    # Ingest should ignore these files
    pandas_df_part1.to_csv(dataset_path / "df1.csv")
    with open(dataset_path / "README", "w") as f:
        f.write("Interesting README content")

    IngestStep.from_pipeline_config(
        pipeline_config={
            "data": {
                "format": "parquet",
                "location": str(dataset_path),
            }
        },
        pipeline_root=os.getcwd(),
    ).run(output_directory=tmp_path)

    reloaded_df = pd.read_parquet(str(tmp_path / "dataset.parquet"))
    pd.testing.assert_frame_equal(reloaded_df, pandas_df)


@pytest.mark.usefixtures("enter_ingest_test_pipeline_directory")
def test_ingest_produces_expected_step_card(pandas_df, tmp_path):
    with mock.patch("shutil.which") as patched_which:
        patched_which.return_value(None)
        dataset_path = tmp_path / "df.parquet"
        pandas_df.to_parquet(dataset_path)

        IngestStep.from_pipeline_config(
            pipeline_config={
                "data": {
                    "format": "parquet",
                    "location": str(dataset_path),
                }
            },
            pipeline_root=os.getcwd(),
        ).run(output_directory=tmp_path)

        expected_step_card_path = os.path.join(tmp_path, "card.html")
        assert os.path.exists(expected_step_card_path)
        with open(expected_step_card_path, "r") as f:
            step_card_html_content = f.read()

        assert "Dataset source location" in step_card_html_content
        assert "Ingested dataset path" in step_card_html_content
        assert "Profile of Ingested Dataset" in step_card_html_content


@pytest.mark.usefixtures("enter_ingest_test_pipeline_directory")
def test_ingest_run_and_inspect_return_expected_step_card(pandas_df, tmp_path):
    dataset_path = tmp_path / "df.parquet"
    pandas_df.to_parquet(dataset_path)

    ingest_step = IngestStep.from_pipeline_config(
        pipeline_config={
            "data": {
                "format": "parquet",
                "location": str(dataset_path),
            }
        },
        pipeline_root=os.getcwd(),
    )

    run_output = ingest_step.run(output_directory=tmp_path)
    inspect_output = ingest_step.inspect(output_directory=tmp_path)

    step_card_output_path = os.path.join(tmp_path, "card.html")
    with open(step_card_output_path, "r") as f:
        step_card_html_content = f.read()

    assert isinstance(run_output, IngestCard)
    assert isinstance(inspect_output, IngestCard)
    assert run_output.to_html() == inspect_output.to_html()
    assert run_output.to_html() == step_card_html_content


@pytest.mark.usefixtures("enter_ingest_test_pipeline_directory")
def test_ingest_throws_when_spark_unavailable_for_spark_based_dataset(spark_df, tmp_path):
    dataset_path = tmp_path / "test.delta"
    spark_df.write.format("delta").save(str(dataset_path))

    with mock.patch(
        "mlflow.pipelines.regression.v1.steps.ingest.datasets._get_active_spark_session",
        side_effect=Exception("Spark unavailable"),
    ), pytest.raises(
        MlflowException, match="Encountered an error while searching for an active Spark session"
    ):
        IngestStep.from_pipeline_config(
            pipeline_config={
                "data": {
                    "format": "delta",
                    "location": str(dataset_path),
                }
            },
            pipeline_root=os.getcwd(),
        ).run(output_directory=tmp_path)


@pytest.mark.usefixtures("enter_ingest_test_pipeline_directory")
def test_ingest_throws_when_dataset_format_unspecified():
    with pytest.raises(MlflowException, match="Dataset format must be specified"):
        IngestStep.from_pipeline_config(
            pipeline_config={
                "data": {
                    "location": "my_location",
                }
            },
            pipeline_root=os.getcwd(),
        )


@pytest.mark.usefixtures("enter_ingest_test_pipeline_directory")
def test_ingest_throws_when_data_section_unspecified():
    with pytest.raises(MlflowException, match="The `data` section.*must be specified"):
        IngestStep.from_pipeline_config(
            pipeline_config={},
            pipeline_root=os.getcwd(),
        )


@pytest.mark.usefixtures("enter_ingest_test_pipeline_directory")
def test_ingest_throws_when_required_dataset_config_keys_are_missing():
    with pytest.raises(MlflowException, match="The `location` configuration key must be specified"):
        IngestStep.from_pipeline_config(
            pipeline_config={
                "data": {
                    "format": "parquet",
                    # Missing location
                }
            },
            pipeline_root=os.getcwd(),
        )

    with pytest.raises(MlflowException, match="The `sql` configuration key must be specified"):
        IngestStep.from_pipeline_config(
            pipeline_config={
                "data": {
                    "format": "spark_sql",
                    # Missing sql
                }
            },
            pipeline_root=os.getcwd(),
        )

    with pytest.raises(
        MlflowException, match="The `custom_loader_method` configuration key must be specified"
    ):
        IngestStep.from_pipeline_config(
            pipeline_config={
                "data": {
                    "format": "csv",
                    "location": "my/dataset.csv",
                    # Missing custom_loader_method
                }
            },
            pipeline_root=os.getcwd(),
        )


@pytest.mark.usefixtures("enter_ingest_test_pipeline_directory")
def test_ingest_throws_when_dataset_files_have_wrong_format(pandas_df, tmp_path):
    dataset_path = tmp_path / "df.csv"
    pandas_df.to_csv(dataset_path)

    with pytest.raises(
        MlflowException, match="Resolved data file.*does not have the expected format"
    ):
        IngestStep.from_pipeline_config(
            pipeline_config={
                "data": {
                    # Intentionally use an incorrect format that doesn't match the dataset
                    "format": "parquet",
                    "location": str(dataset_path),
                }
            },
            pipeline_root=os.getcwd(),
        ).run(output_directory=tmp_path)

    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    pandas_df_part1 = pandas_df[:1]
    pandas_df_part2 = pandas_df[1:]
    pandas_df_part1.to_csv(dataset_path / "df1.csv")
    pandas_df_part2.to_csv(dataset_path / "df2.csv")

    with pytest.raises(
        MlflowException, match="Did not find any data files with the specified format"
    ):
        IngestStep.from_pipeline_config(
            pipeline_config={
                "data": {
                    # Intentionally use an incorrect format that doesn't match the dataset
                    "format": "parquet",
                    "location": str(dataset_path),
                }
            },
            pipeline_root=os.getcwd(),
        ).run(output_directory=tmp_path)
