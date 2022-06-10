import os
import pathlib
import shutil
import hashlib

import pytest

from mlflow.exceptions import MlflowException
from mlflow.pipelines.utils import (
    get_pipeline_root_path,
    get_hashed_pipeline_root,
    get_pipeline_name,
    get_pipeline_config,
    get_default_profile,
    display_html,
)
from mlflow.utils.file_utils import write_yaml

# pylint: disable=unused-import
from tests.pipelines.helper_functions import (
    enter_pipeline_example_directory,
    enter_test_pipeline_directory,
)  # pylint: enable=unused-import
from tests.pipelines.helper_functions import chdir
from unittest import mock


def test_get_pipeline_root_path_returns_correctly_when_inside_pipeline_directory(
    enter_pipeline_example_directory,
):
    pipeline_root_path = enter_pipeline_example_directory
    assert get_pipeline_root_path() == str(pipeline_root_path)
    os.chdir(pathlib.Path.cwd() / "notebooks")
    assert get_pipeline_root_path() == str(enter_pipeline_example_directory)


def test_get_pipeline_root_path_throws_outside_pipeline_directory(tmp_path):
    with pytest.raises(MlflowException, match="Failed to find pipeline.yaml"), chdir(tmp_path):
        get_pipeline_root_path()


def test_get_pipeline_name_returns_correctly_for_valid_pipeline_directory(
    enter_pipeline_example_directory, tmp_path
):
    pipeline_root_path = enter_pipeline_example_directory
    assert pathlib.Path.cwd() == pipeline_root_path
    assert get_pipeline_name() == "sklearn_regression"

    with chdir(tmp_path):
        assert get_pipeline_name(pipeline_root_path=pipeline_root_path) == "sklearn_regression"


def test_get_hashed_pipeline_root_returns_correctly_for_valid_pipeline_directory(
    enter_pipeline_example_directory, tmp_path
):
    pipeline_root_path = enter_pipeline_example_directory
    pipeline_root_path_str = str(pipeline_root_path)
    assert pathlib.Path.cwd() == pipeline_root_path
    assert (
        get_hashed_pipeline_root()
        == hashlib.sha256(pipeline_root_path_str.encode("utf-8")).hexdigest()
    )

    with chdir(tmp_path):
        assert (
            get_hashed_pipeline_root(pipeline_root_path=pipeline_root_path_str)
            == hashlib.sha256(pipeline_root_path_str.encode("utf-8")).hexdigest()
        )


def test_get_pipeline_name_throws_for_invalid_pipeline_directory(tmp_path):
    with pytest.raises(MlflowException, match="Failed to find pipeline.yaml"), chdir(tmp_path):
        get_pipeline_name()

    with pytest.raises(MlflowException, match="Failed to find pipeline.yaml"):
        get_pipeline_name(pipeline_root_path=tmp_path)


def test_get_pipeline_config_returns_correctly_for_valid_pipeline_directory(
    enter_pipeline_example_directory, tmp_path
):
    pipeline_root_path = enter_pipeline_example_directory
    test_pipeline_root_path = tmp_path / "test_pipeline"
    shutil.copytree(pipeline_root_path, test_pipeline_root_path)

    test_pipeline_config = {
        "config1": 10,
        "config2": {
            "subconfig": ["A"],
        },
        "config3": "3",
    }
    write_yaml(test_pipeline_root_path, "pipeline.yaml", test_pipeline_config, overwrite=True)

    with chdir(test_pipeline_root_path):
        assert pathlib.Path.cwd() == test_pipeline_root_path
        assert get_pipeline_config() == test_pipeline_config

    with chdir(tmp_path):
        assert (
            get_pipeline_config(pipeline_root_path=test_pipeline_root_path) == test_pipeline_config
        )


def test_get_pipeline_config_throws_for_invalid_pipeline_directory(tmp_path):
    with pytest.raises(MlflowException, match="Failed to find pipeline.yaml"), chdir(tmp_path):
        get_pipeline_config()

    with pytest.raises(MlflowException, match="Failed to find pipeline.yaml"):
        get_pipeline_config(pipeline_root_path=tmp_path)


@pytest.mark.usefixtures("enter_test_pipeline_directory")
def test_get_pipeline_config_supports_empty_profile():
    with open("profiles/empty.yaml", "w"):
        pass

    get_pipeline_config(profile="empty")


@pytest.mark.usefixtures("enter_pipeline_example_directory")
def test_get_pipeline_config_throws_for_nonexistent_profile():
    with pytest.raises(MlflowException, match="Yaml file.*badprofile.*does not exist"):
        get_pipeline_config(profile="badprofile")


def test_get_default_profile_works():
    assert get_default_profile() == "local"
    with mock.patch(
        "mlflow.pipelines.utils.is_in_databricks_runtime", return_value=True
    ) as patched_is_in_databricks_runtime:
        assert get_default_profile() == "databricks"
        patched_is_in_databricks_runtime.assert_called_once()


def test_display_html_raises_without_input():
    with pytest.raises(MlflowException, match="At least one HTML source must be provided"):
        display_html()


def test_display_html_opens_html_data():
    html_data = "<!DOCTYPE html><html><body><p>Hey</p></body></html>"
    with mock.patch("mlflow.pipelines.utils.is_running_in_ipython_environment", return_value=True):
        with mock.patch("IPython.display.display") as patched_display:
            display_html(html_data=html_data)
            patched_display.assert_called_once()


def test_display_html_opens_html_file(tmp_path):
    html_file = tmp_path / "test.html"
    html_file.write_text("<!DOCTYPE html><html><body><p>Hey</p></body></html>")
    with mock.patch("subprocess.run") as patched_subprocess, mock.patch(
        "shutil.which", return_value=True
    ):
        display_html(html_file_path=html_file)
        patched_subprocess.assert_called_once()
