import os
from mlflow.utils.file_utils import chdir, read_yaml, write_yaml
from mlflow.utils.process import _exec_cmd


_MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR = "MLFLOW_PIPELINES_EXECUTION_DIRECTORY"
_STEP_OUTPUTS_SUBDIRECTORY_NAME = "outputs"
_STEP_CONF_YAML_NAME = "conf.yaml"


def run_step(pipeline_root_path, pipeline_name, pipeline_steps, target_step):
    execution_dir_path = _get_or_create_execution_directory(pipeline_root_path, pipeline_name, pipeline_steps)
    _write_updated_step_confs(execution_directory_path=execution_dir_path, pipeline_root_path=pipeline_root_path, pipeline_steps=pipeline_steps)
    with chdir(execution_dir_path):
        _run_make(target_step)
    return _get_step_output_directory_path(execution_directory_path=execution_dir_path, step_name=target_step)


def _get_or_create_execution_directory(pipeline_root_path, pipeline_name, pipeline_steps):
    execution_dir_path = (
        os.environ.get(_MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR)
        or os.path.join(os.path.expanduser("~"), ".mlflow", "pipelines", pipeline_name)
    )

    os.makedirs(execution_dir_path, exist_ok=True)
    _create_makefile(pipeline_root_path, execution_dir_path)
    for step_name in pipeline_steps:
        step_output_subdir_path = _get_step_output_directory_path(execution_dir_path, step_name)
        os.makedirs(step_output_subdir_path, exist_ok=True)

    return execution_dir_path


def _write_updated_step_confs(execution_directory_path, pipeline_root_path, pipeline_steps):
    for step_name in pipeline_steps:
        step_subdir_path = os.path.join(execution_directory_path, step_name)
        step_conf_path = os.path.join(execution_directory_path, step_name, _STEP_CONF_YAML_NAME)
        if os.path.exists(step_conf_path):
            prev_step_conf = read_yaml(root=step_subdir_path, file_name=_STEP_CONF_YAML_NAME)
        else:
            prev_step_conf = None

        # TODO: Extract the conf from the pipeline step
        step_conf = {
            "pipeline_root": pipeline_root_path,
        }

        if prev_step_conf != step_conf:
            write_yaml(root=step_subdir_path, file_name=_STEP_CONF_YAML_NAME, data=step_conf, overwrite=True, sort_keys=True)


def _get_step_output_directory_path(execution_directory_path, step_name):
    return os.path.abspath(os.path.join(execution_directory_path, step_name, _STEP_OUTPUTS_SUBDIRECTORY_NAME))


def _run_make(rule_name):
    _exec_cmd(["make", rule_name], stream_output=True, synchronous=True)


def _create_makefile(pipeline_root_path, execution_directory_path):
    makefile_path = os.path.join(execution_directory_path, "Makefile")
    with open(makefile_path, "w") as f:
        f.write(_MAKEFILE_FORMAT_STRING.format(prp=os.path.abspath(pipeline_root_path)))


_MAKEFILE_FORMAT_STRING = """\
split_objects = split/outputs/train.parquet split/outputs/test.parquet split/outputs/summary.html

split: $(split_objects)

%/outputs/train.parquet %/outputs/test.parquet %/outputs/summary.html: {prp}/datasets/autos.parquet
	python -c "from mlflow.pipelines.split_step import run_split_step; run_split_step(input_path='{prp}/datasets/autos.parquet', summary_path='$*/outputs/summary.html', train_output_path='$*/outputs/train.parquet', test_output_path='$*/outputs/test.parquet')"

transform_objects = transform/outputs/transformer.pkl transform/outputs/train_transformed.parquet

transform: $(transform_objects)

%/outputs/transformer.pkl %/outputs/train_transformed.parquet: {prp}/steps/transform.py {prp}/steps/transformer_config.yaml split/outputs/train.parquet transform/conf.yaml
	python -c "from mlflow.pipelines.transform_step import run_transform_step; run_transform_step(train_data_path='split/outputs/train.parquet', transformer_config_path='{prp}/steps/transformer_config.yaml', transformer_output_path='$*/outputs/transformer.pkl', transformed_data_output_path='$*/outputs/train_transformed.parquet', step_config_path='transform/conf.yaml')"

train_objects = train_pipeline.pkl train_run_id

train: $(train_objects)

%_pipeline.pkl %_run_id: {prp}/steps/train.py {prp}/steps/train_config.yaml transform_train_transformed.parquet transform_transformer.pkl
	python -c "from mlflow.pipelines.train_step import run_train_step; run_train_step(transformed_train_data_path='transform_train_transformed.parquet', train_config_path='{prp}/steps/train_config.yaml', transformer_path='transform_transformer.pkl', tracking_uri='file:/tmp/mlruns', pipeline_output_path='$*_pipeline.pkl', run_id_output_path='$*_run_id')"

evaluate_objects = evaluate_worst_training_examples.parquet evaluate_metrics.json evaluate_explanations.html

evaluate: $(evaluate_objects)

%_worst_training_examples.parquet %_metrics.json %_explanations.html: train_pipeline.pkl split_train.parquet train_run_id
	python -c "from mlflow.pipelines.evaluate_step import run_evaluate_step; run_evaluate_step(pipeline_path='train_pipeline.pkl', tracking_uri='file:/tmp/mlruns', run_id_path='train_run_id', train_data_path='split_train.parquet', test_data_path='split_test.parquet', explanations_output_path='$*_explanations.html', metrics_output_path='$*_metrics.json', worst_train_examples_output_path='$*_worst_training_examples.parquet')"

clean:
	rm -rf $(split_objects) $(transform_objects) $(train_objects) $(evaluate_objects)
"""
