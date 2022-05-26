import pathlib
import shutil
from unittest import mock

import pytest

from mlflow.pipelines.utils.execution import _get_or_create_execution_directory, run_pipeline_step

from tests.pipelines.helper_functions import BaseStepImplemented


def test_get_or_create_execution_directory_is_idempotent(tmp_path):
    class TestStep(BaseStepImplemented):
        def __init__(self):  # pylint: disable=super-init-not-called
            pass

        @property
        def name(self):
            return "test_step"

    test_step = TestStep()

    def assert_expected_execution_directory_contents_exist(execution_dir_path):
        assert (execution_dir_path / "Makefile").exists()
        assert (execution_dir_path / "steps").exists()
        assert (execution_dir_path / "steps" / test_step.name / "outputs").exists()

    execution_dir_path_1 = pathlib.Path(
        _get_or_create_execution_directory(
            pipeline_root_path=tmp_path, pipeline_name="test_pipeline", pipeline_steps=[test_step]
        )
    )
    execution_dir_path_2 = pathlib.Path(
        _get_or_create_execution_directory(
            pipeline_root_path=tmp_path, pipeline_name="test_pipeline", pipeline_steps=[test_step]
        )
    )
    assert execution_dir_path_1 == execution_dir_path_2
    assert_expected_execution_directory_contents_exist(execution_dir_path_1)

    shutil.rmtree(execution_dir_path_1)

    # Simulate a failure with Makefile creation
    with mock.patch(
        "mlflow.pipelines.utils.execution._create_makefile",
        side_effect=Exception("Makefile creation failed"),
    ), pytest.raises(Exception, match="Makefile creation failed"):
        _get_or_create_execution_directory(
            pipeline_root_path=tmp_path, pipeline_name="test_pipeline", pipeline_steps=[test_step]
        )

    # Verify that the directory exists but is empty due to short circuiting after
    # failed Makefile creation
    assert execution_dir_path_1.exists()
    assert next(execution_dir_path_1.iterdir(), None) == None

    # Re-create the execution directory and verify that all expected contents are present
    execution_dir_path_3 = pathlib.Path(
        _get_or_create_execution_directory(
            pipeline_root_path=tmp_path, pipeline_name="test_pipeline", pipeline_steps=[test_step]
        )
    )
    assert execution_dir_path_3 == execution_dir_path_1
    assert_expected_execution_directory_contents_exist(execution_dir_path_3)

    shutil.rmtree(execution_dir_path_1)

    # Simulate a failure with step-specific directory creation
    with mock.patch(
        "mlflow.pipelines.utils.execution._get_step_output_directory_path",
        side_effect=Exception("Step directory creation failed"),
    ), pytest.raises(Exception, match="Step directory creation failed"):
        _get_or_create_execution_directory(
            pipeline_root_path=tmp_path, pipeline_name="test_pipeline", pipeline_steps=[test_step]
        )

    # Verify that the directory exists & that a Makefile is present but step-specific directories
    # were not created due to failures
    assert execution_dir_path_1.exists()
    assert [path.name for path in execution_dir_path_1.iterdir()] == ["Makefile"]

    # Re-create the execution directory and verify that all expected contents are present
    execution_dir_path_4 = pathlib.Path(
        _get_or_create_execution_directory(
            pipeline_root_path=tmp_path, pipeline_name="test_pipeline", pipeline_steps=[test_step]
        )
    )
    assert execution_dir_path_4 == execution_dir_path_1
    assert_expected_execution_directory_contents_exist(execution_dir_path_4)


def test_run_pipeline_step_sets_environment_as_expected(tmp_path):
    class TestStep1(BaseStepImplemented):
        def __init__(self):  # pylint: disable=super-init-not-called
            self.step_config = {}

        @property
        def name(self):
            return "test_step_1"

        @property
        def environment(self):
            return {"A": "B"}

    class TestStep2(BaseStepImplemented):
        def __init__(self):  # pylint: disable=super-init-not-called
            self.step_config = {}

        @property
        def name(self):
            return "test_step_2"

        @property
        def environment(self):
            return {"C": "D"}

    with mock.patch("mlflow.pipelines.utils.execution._exec_cmd") as mock_run_in_subprocess:
        run_pipeline_step(
            pipeline_root_path=tmp_path,
            pipeline_name="test_pipeline",
            pipeline_steps=[TestStep1(), TestStep2()],
            target_step=TestStep1(),
        )

    extra_env = mock_run_in_subprocess.call_args.kwargs.get("extra_env")
    print("EXTRA ENV", extra_env)
    assert extra_env == {"A": "B", "C": "D"}, extra_env
