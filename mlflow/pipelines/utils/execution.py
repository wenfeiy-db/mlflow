import os
import pathlib
import shutil
from typing import List, Dict

from mlflow.pipelines.step import BaseStep
from mlflow.utils.file_utils import read_yaml, write_yaml
from mlflow.utils.process import _exec_cmd


_MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR = "MLFLOW_PIPELINES_EXECUTION_DIRECTORY"
_STEPS_SUBDIRECTORY_NAME = "steps"
_STEP_OUTPUTS_SUBDIRECTORY_NAME = "outputs"
_STEP_CONF_YAML_NAME = "conf.yaml"


def run_pipeline_step(
    pipeline_root_path: str,
    pipeline_name: str,
    pipeline_steps: List[BaseStep],
    target_step: BaseStep,
) -> str:
    """
    Runs the specified step in the specified pipeline, as well as all dependent steps.

    :param pipeline_root_path: The absolute path of the pipeline root directory on the local
                               filesystem.
    :param pipeline_name: The name of the pipeline.
    :param pipeline_steps: A list of all the steps contained in the specified pipeline.
    :param target_step: The step to run.
    :return: The absolute path of the step's execution outputs on the local filesystem.
    """
    execution_dir_path = _get_or_create_execution_directory(
        pipeline_root_path, pipeline_name, pipeline_steps
    )
    _write_updated_step_confs(
        pipeline_steps=pipeline_steps,
        execution_directory_path=execution_dir_path,
    )
    step_output_directory_path = _get_step_output_directory_path(
        execution_directory_path=execution_dir_path, step_name=target_step.name
    )
    # Aggregate step-specific environment variables into a single environment dictionary
    # that is passed to the Make subprocess. In the future, steps with different environments
    # should be isolated in different subprocesses
    make_env = {}
    for step in pipeline_steps:
        make_env.update(step.environment)
    _run_make(
        execution_directory_path=execution_dir_path,
        rule_name=target_step.name,
        extra_env=make_env,
    )
    return step_output_directory_path


def clean_execution_state(pipeline_name: str, pipeline_steps: List[BaseStep]) -> None:
    """
    Removes all execution state for the specified pipeline from the associated execution directory
    on the local filesystem. This method does *not* remove other execution results, such as content
    logged to MLflow Tracking.

    :param pipeline_name: The name of the pipeline.
    """
    execution_dir_path = get_execution_directory_path(pipeline_name=pipeline_name)
    for step in pipeline_steps:
        step_outputs_path = _get_step_output_directory_path(
            execution_directory_path=execution_dir_path,
            step_name=step.name,
        )
        if os.path.exists(step_outputs_path):
            shutil.rmtree(step_outputs_path)


def get_step_output_path(pipeline_name: str, step_name: str, relative_path: str) -> str:
    """
    Obtains the absolute path of the specified step output on the local filesystem. Does
    not check the existence of the output.

    :param pipeline_name: The name of the pipeline.
    :param step_name: The name of the pipeline step containing the specified output.
    :param relative_path: The relative path of the output within the output directory
                          of the specified pipeline step.
    :return The absolute path of the step output on the local filesystem, which may or may
            not exist.
    """
    execution_dir_path = get_execution_directory_path(pipeline_name=pipeline_name)
    step_outputs_path = _get_step_output_directory_path(
        execution_directory_path=execution_dir_path,
        step_name=step_name,
    )
    return os.path.abspath(os.path.join(step_outputs_path, relative_path))


def _get_or_create_execution_directory(
    pipeline_root_path: str, pipeline_name: str, pipeline_steps: List[BaseStep]
) -> str:
    """
    Obtains the path of the execution directory on the local filesystem corresponding to the
    specified pipeline, creating the execution directory and its required contents if they do
    not already exist.

    :param pipeline_root_path: The absolute path of the pipeline root directory on the local
                               filesystem.
    :param pipeline_name: The name of the pipeline.
    :param pipeline_steps: A list of all the steps contained in the specified pipeline.
    :return: The absolute path of the execution directory on the local filesystem for the specified
             pipeline.
    """
    execution_dir_path = get_execution_directory_path(pipeline_name)

    os.makedirs(execution_dir_path, exist_ok=True)
    _create_makefile(pipeline_root_path, execution_dir_path)
    for step in pipeline_steps:
        step_output_subdir_path = _get_step_output_directory_path(execution_dir_path, step.name)
        os.makedirs(step_output_subdir_path, exist_ok=True)

    return execution_dir_path


def _write_updated_step_confs(
    pipeline_steps: List[BaseStep], execution_directory_path: str
) -> None:
    """
    Compares the in-memory configuration state of the specified pipeline steps with step-specific
    internal configuration files written by prior executions. If updates are found, writes updated
    state to the corresponding files. If no updates are found, configuration state is not
    rewritten.

    :param pipeline_steps: A list of all the steps contained in the specified pipeline.
    :param execution_directory_path: The absolute path of the execution directory on the local
                                     filesystem for the specified pipeline. Configuration files are
                                     written to step-specific subdirectories of this execution
                                     directory.
    """
    for step in pipeline_steps:
        step_subdir_path = os.path.join(
            execution_directory_path, _STEPS_SUBDIRECTORY_NAME, step.name
        )
        step_conf_path = os.path.join(step_subdir_path, _STEP_CONF_YAML_NAME)
        if os.path.exists(step_conf_path):
            prev_step_conf = read_yaml(root=step_subdir_path, file_name=_STEP_CONF_YAML_NAME)
        else:
            prev_step_conf = None

        if prev_step_conf != step.step_config:
            write_yaml(
                root=step_subdir_path,
                file_name=_STEP_CONF_YAML_NAME,
                data=step.step_config,
                overwrite=True,
                sort_keys=True,
            )


def get_execution_directory_path(pipeline_name: str) -> str:
    """
    Obtains the path of the execution directory on the local filesystem corresponding to the
    specified pipeline, which may or may not exist.

    :param pipeline_name: The name of the pipeline for which to obtain the associated execution
                          directory path.
    """
    return os.path.abspath(
        os.environ.get(_MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR)
        or os.path.join(os.path.expanduser("~"), ".mlflow", "pipelines", pipeline_name)
    )


def _get_step_output_directory_path(execution_directory_path: str, step_name: str) -> str:
    """
    Obtains the path of the local filesystem directory containing outputs for the specified step,
    which may or may not exist.

    :param execution_directory_path: The absolute path of the execution directory on the local
                                     filesystem for the relevant pipeline. The Makefile is created
                                     in this directory.
    :param step_name: The name of the pipeline step for which to obtain the output directory path.
    :return The absolute path of the local filesystem directory containing outputs for the specified
            step.
    """
    return os.path.abspath(
        os.path.join(
            execution_directory_path,
            _STEPS_SUBDIRECTORY_NAME,
            step_name,
            _STEP_OUTPUTS_SUBDIRECTORY_NAME,
        )
    )


def _run_make(execution_directory_path, rule_name: str, extra_env: Dict[str, str]) -> None:
    """
    Runs the specified pipeline rule with Make. This method assumes that a Makefile named `Makefile`
    exists in the specified execution directory.

    :param execution_directory_path: The absolute path of the execution directory on the local
                                     filesystem for the relevant pipeline. The Makefile is created
                                     in this directory.
    :param extra_env: Extra environment variables to be defined when running the Make child process.
    :param rule_name: The name of the Make rule to run.
    """
    _exec_cmd(
        ["make", "-f", "Makefile", rule_name],
        capture_output=False,
        stream_output=True,
        synchronous=True,
        cwd=execution_directory_path,
        extra_env=extra_env,
    )


def _create_makefile(pipeline_root_path, execution_directory_path) -> None:
    """
    Creates a Makefile with a set of relevant MLflow Pipelines targets for the specified pipeline,
    overwriting the preexisting Makefile if one exists. The Makefile is created in the specified
    execution directory.

    :param pipeline_root_path: The absolute path of the pipeline root directory on the local
                               filesystem.
    :param execution_directory_path: The absolute path of the execution directory on the local
                                     filesystem for the specified pipeline. The Makefile is created
                                     in this directory.
    """
    makefile_path = os.path.join(execution_directory_path, "Makefile")
    makefile_contents = _MAKEFILE_FORMAT_STRING.format(
        path=_MakefilePathFormat(
            os.path.abspath(pipeline_root_path),
            execution_directory_path=os.path.abspath(execution_directory_path),
        ),
    )
    with open(makefile_path, "w") as f:
        f.write(makefile_contents)


class _MakefilePathFormat:
    r"""
    Provides platform-agnostic path substitution for execution Makefiles, ensuring that POSIX-style
    relative paths are joined correctly with POSIX-style or Windows-style pipeline root paths.

    For example, given a format string `s = "{path:prp/my/subpath.txt}"`, invoking
    `s.format(path=_MakefilePathFormat(pipeline_root_path="/my/pipeline/root/path", ...))` on
    Unix systems or
    `s.format(path=_MakefilePathFormat(pipeline_root_path="C:\my\pipeline\root\path", ...))`` on
    Windows systems will yield "/my/pipeline/root/path/my/subpath.txt" or
    "C:/my/pipeline/root/path/my/subpath.txt", respectively.

    Additionally, given a format string `s = "{path:exe/my/subpath.txt}"`, invoking
    `s.format(path=_MakefilePathFormat(execution_directory_path="/my/exe/dir/path", ...))` on
    Unix systems or
    `s.format(path=_MakefilePathFormat(execution_directory_path="/my/exe/dir/path", ...))`` on
    Windows systems will yield "/my/exe/dir/path/my/subpath.txt" or
    "C:/my/exe/dir/path/my/subpath.txt", respectively.
    """

    _PIPELINE_ROOT_PATH_PREFIX_PLACEHOLDER = "prp/"
    _EXECUTION_DIRECTORY_PATH_PREFIX_PLACEHOLDER = "exe/"

    def __init__(self, pipeline_root_path: str, execution_directory_path: str):
        """
        :param pipeline_root_path: The absolute path of the pipeline root directory on the local
                                   filesystem.
        :param execution_directory_path: The absolute path of the execution directory on the local
                                         filesystem for the pipeline.
        """
        self.pipeline_root_path = pipeline_root_path
        self.execution_directory_path = execution_directory_path

    def _get_formatted_path(
        self, path_spec: str, prefix_placeholder: str, replacement_path: str
    ) -> str:
        """
        :param path_spec: A substitution path spec of the form `<placeholder>/<subpath>`. This
                          method substitutes `<placeholder>` with `<pipeline_root_path>`, if
                          `<placeholder>` is `prp`, or `<execution_directory_path>`, if
                          `<placeholder>` is `exe`.
        :param prefix_placeholder: The prefix placeholder, which is present at the beginning of
                                   `path_spec`. Either `prp` or `exe`.
        :param replacement_path: The path to use to replace the specified `prefix_placeholder`
                                 in the specified `path_spec`.
        :return: The formatted path obtained by replacing the ``prefix placeholder`` in the
                 specified ``path_spec`` with the specified ``replacement_path``.
        """
        subpath = pathlib.PurePosixPath(path_spec.split(prefix_placeholder)[1])
        pipeline_root_posix_path = pathlib.PurePosixPath(pathlib.Path(replacement_path).as_posix())
        full_formatted_path = pipeline_root_posix_path / subpath
        return str(full_formatted_path)

    def __format__(self, path_spec: str) -> str:
        """
        :param path_spec: A substitution path spec of the form `<placeholder>/<subpath>`. This
                          method substitutes `<placeholder>` with `<pipeline_root_path>`, if
                          `<placeholder>` is `prp`, or `<execution_directory_path>`, if
                          `<placeholder>` is `exe`.
        """
        if path_spec.startswith(_MakefilePathFormat._PIPELINE_ROOT_PATH_PREFIX_PLACEHOLDER):
            return self._get_formatted_path(
                path_spec=path_spec,
                prefix_placeholder=_MakefilePathFormat._PIPELINE_ROOT_PATH_PREFIX_PLACEHOLDER,
                replacement_path=self.pipeline_root_path,
            )
        elif path_spec.startswith(_MakefilePathFormat._EXECUTION_DIRECTORY_PATH_PREFIX_PLACEHOLDER):
            return self._get_formatted_path(
                path_spec=path_spec,
                prefix_placeholder=_MakefilePathFormat._EXECUTION_DIRECTORY_PATH_PREFIX_PLACEHOLDER,
                replacement_path=self.execution_directory_path,
            )
        else:
            raise ValueError(f"Invalid Makefile string format path spec: {path_spec}")


# Makefile contents for cache-aware pipeline execution. These contents include variable placeholders
# that need to be formatted (substituted) with the pipeline root directory in order to produce a
# valid Makefile
_MAKEFILE_FORMAT_STRING = r"""
# Define `ingest` as a target with no dependencies to ensure that it runs whenever a user explicitly
# invokes the MLflow Pipelines ingest step, allowing them to reingest data on-demand
ingest:
	cd {path:prp/} && \
        python -c "from mlflow.pipelines.regression.v1.steps.ingest import IngestStep; IngestStep.from_step_config_path(step_config_path='{path:exe/steps/ingest/conf.yaml}', pipeline_root='{path:prp/}').run(output_directory='{path:exe/steps/ingest/outputs}')"

# Define a separate target for the ingested dataset that recursively invokes make with the `ingest`
# target. Downstream steps depend on the ingested dataset target, rather than the `ingest` target,
# ensuring that data is only ingested for downstream steps if it is not already present on the
# local filesystem
steps/ingest/outputs/dataset.parquet: steps/ingest/conf.yaml
	$(MAKE) ingest

split_objects = steps/split/outputs/train.parquet steps/split/outputs/test.parquet steps/split/outputs/summary.html

split: $(split_objects)

steps/%/outputs/train.parquet steps/%/outputs/test.parquet steps/%/outputs/summary.html: steps/ingest/outputs/dataset.parquet steps/split/conf.yaml
	cd {path:prp/} && \
        python -c "from mlflow.pipelines.regression.v1.steps.split import SplitStep; SplitStep.from_step_config_path(step_config_path='{path:exe/steps/split/conf.yaml}', pipeline_root='{path:prp/}').run(output_directory='{path:exe/steps/split/outputs}')"

transform_objects = steps/transform/outputs/transformer.pkl steps/transform/outputs/train_transformed.parquet

transform: $(transform_objects)

steps/%/outputs/transformer.pkl steps/%/outputs/train_transformed.parquet: {path:prp/steps/transform.py} {path:prp/steps/transformer_config.yaml} steps/split/outputs/train.parquet steps/transform/conf.yaml
	cd {path:prp/} && \
        python -c "from mlflow.pipelines.regression.v1.steps.transform import TransformStep; TransformStep.from_step_config_path(step_config_path='{path:exe/steps/transform/conf.yaml}', pipeline_root='{path:prp/}').run(output_directory='{path:exe/steps/transform/outputs}')"

train_objects = steps/train/outputs/pipeline.pkl steps/train/outputs/run_id

train: $(train_objects)

steps/%/outputs/pipeline.pkl steps/%/outputs/run_id: {path:prp/steps/train.py} {path:prp/steps/train_config.yaml} steps/transform/outputs/train_transformed.parquet steps/transform/outputs/transformer.pkl steps/train/conf.yaml
	cd {path:prp/} && \
        python -c "from mlflow.pipelines.regression.v1.steps.train import TrainStep; TrainStep.from_step_config_path(step_config_path='{path:exe/steps/train/conf.yaml}', pipeline_root='{path:prp/}').run(output_directory='{path:exe/steps/train/outputs}')"

evaluate_objects = steps/evaluate/outputs/worst_training_examples.parquet steps/evaluate/outputs/metrics.json steps/evaluate/outputs/explanations.html steps/evaluate/outputs/model_validation_status

evaluate: $(evaluate_objects)

steps/%/outputs/worst_training_examples.parquet steps/%/outputs/metrics.json steps/%/outputs/explanations.html steps/%/outputs/model_validation_status: steps/train/outputs/pipeline.pkl steps/split/outputs/train.parquet steps/split/outputs/test.parquet steps/train/outputs/run_id steps/evaluate/conf.yaml
	cd {path:prp/} && \
        python -c "from mlflow.pipelines.regression.v1.steps.evaluate import EvaluateStep; EvaluateStep.from_step_config_path(step_config_path='{path:exe/steps/evaluate/conf.yaml}', pipeline_root='{path:prp/}').run(output_directory='{path:exe/steps/evaluate/outputs}')"

register_objects = steps/register/outputs/register_card.html

register: $(register_objects)

steps/register/outputs/register_card.html: steps/train/outputs/run_id steps/register/conf.yaml steps/evaluate/outputs/model_validation_status
	python -c "from mlflow.pipelines.regression.v1.steps.register import RegisterStep; RegisterStep.from_step_config_path(step_config_path='steps/register/conf.yaml', pipeline_root='{path:prp/}').run(output_directory='steps/register/outputs')"

clean:
	rm -rf $(split_objects) $(transform_objects) $(train_objects) $(evaluate_objects)
"""
