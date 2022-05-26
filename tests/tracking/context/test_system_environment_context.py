import os

import pytest

from mlflow.tracking.context.system_environment_context import SystemEnvironmentContext


@pytest.fixture(autouse=True)
def reset_run_context_env_var():
    run_context = os.environ.get("MLFLOW_RUN_CONTEXT")
    try:
        yield
    finally:
        if run_context is not None:
            os.environ["MLFLOW_RUN_CONTEXT"] = run_context
        else:
            os.environ.pop("MLFLOW_RUN_CONTEXT", None)


def test_system_environment_context_in_context():
    os.environ["MLFLOW_RUN_CONTEXT"] = '{"A": "B"}'
    assert SystemEnvironmentContext().in_context()
    del os.environ["MLFLOW_RUN_CONTEXT"]
    assert not SystemEnvironmentContext().in_context()


def test_system_environment_context_tags():
    os.environ["MLFLOW_RUN_CONTEXT"] = '{"A": "B", "C": "D"}'
    assert SystemEnvironmentContext().tags() == {"A": "B", "C": "D"}
