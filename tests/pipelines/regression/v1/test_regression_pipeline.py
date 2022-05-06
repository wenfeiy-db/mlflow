from unittest.mock import patch
from mlflow.pipelines.regression.v1.pipeline import Pipeline


def test_setup_pipeline_initialization():
    pipeline_root = "pipeline-root"
    name = "sklearn_regression"
    pipeline_config = {"template": "regression/v1"}
    profile = "production"

    pipeline = Pipeline(profile, pipeline_root)

    with patch(
        "mlflow.utils.file_utils.render_and_merge_yaml", return_value=pipeline_config
    ) as patch_render_and_merge_yaml, patch(
        "os.path.join", return_value="profiles/production.yaml"
    ) as patch_os_path_join, patch(
        "mlflow.pipelines.regression.v1.pipeline.get_pipeline_name", return_value=name
    ) as patch_get_pipeline_name, patch(
        "mlflow.pipelines.regression.v1.pipeline.IngestStep.from_pipeline_config"
    ) as patch_ingest_step, patch(
        "mlflow.pipelines.regression.v1.pipeline.SplitStep.from_pipeline_config"
    ) as patch_split_step, patch(
        "mlflow.pipelines.regression.v1.pipeline.TransformStep.from_pipeline_config"
    ) as patch_transform_step, patch(
        "mlflow.pipelines.regression.v1.pipeline.TrainStep.from_pipeline_config"
    ) as patch_train_step, patch(
        "mlflow.pipelines.regression.v1.pipeline.EvaluateStep.from_pipeline_config"
    ) as patch_evaluate_step:

        pipeline_name, [
            ingestStep,
            splitStep,
            transformStep,
            trainStep,
            evaluateStep,
        ] = pipeline.resolve_pipeline_steps()
        patch_render_and_merge_yaml.assert_called_once_with(
            pipeline_root, "pipeline.yaml", "profiles/production.yaml"
        )
        patch_os_path_join.assert_called_once_with("profiles", profile + ".yaml")
        patch_get_pipeline_name.assert_called_once()
        patch_ingest_step.assert_called_once_with(pipeline_config, pipeline_root)
        patch_split_step.assert_called_once_with(pipeline_config, pipeline_root)
        patch_transform_step.assert_called_once_with(pipeline_config, pipeline_root)
        patch_train_step.assert_called_once_with(pipeline_config, pipeline_root)
        patch_evaluate_step.assert_called_once_with(pipeline_config, pipeline_root)

        assert pipeline_name == name
        assert ingestStep == patch_ingest_step.return_value
        assert splitStep == patch_split_step.return_value
        assert transformStep == patch_transform_step.return_value
        assert trainStep == patch_train_step.return_value
        assert evaluateStep == patch_evaluate_step.return_value


def test_ingest_step():
    pipeline_root = "pipeline-root"
    pipeline_name = "sklearn_regression"
    profile = "production"
    pipeline_steps = ["ingestStep", "splitStep", "transformStep", "trainStep", "evaluateStep"]
    pipeline = Pipeline(profile, pipeline_root)
    with patch.object(
        pipeline,
        "resolve_pipeline_steps",
        return_value=(pipeline_name, pipeline_steps),
    ), patch(
        "mlflow.pipelines.regression.v1.pipeline.run_pipeline_step"
    ) as patch_run_pipeline_step:
        pipeline.ingest()
        patch_run_pipeline_step.assert_called_once_with(
            pipeline_root, pipeline_name, pipeline_steps, pipeline_steps[0]
        )


def test_split_step():
    pipeline_root = "pipeline-root"
    pipeline_name = "sklearn_regression"
    profile = "production"
    pipeline_steps = ["ingestStep", "splitStep", "transformStep", "trainStep", "evaluateStep"]
    pipeline = Pipeline(profile, pipeline_root)
    with patch.object(
        pipeline,
        "resolve_pipeline_steps",
        return_value=(pipeline_name, pipeline_steps),
    ), patch(
        "mlflow.pipelines.regression.v1.pipeline.run_pipeline_step"
    ) as patch_run_pipeline_step:
        pipeline.split()
        patch_run_pipeline_step.assert_called_once_with(
            pipeline_root, pipeline_name, pipeline_steps, pipeline_steps[1]
        )


def test_transform_step():
    pipeline_root = "pipeline-root"
    pipeline_name = "sklearn_regression"
    profile = "production"
    pipeline_steps = ["ingestStep", "splitStep", "transformStep", "trainStep", "evaluateStep"]
    pipeline = Pipeline(profile, pipeline_root)
    with patch.object(
        pipeline,
        "resolve_pipeline_steps",
        return_value=(pipeline_name, pipeline_steps),
    ), patch(
        "mlflow.pipelines.regression.v1.pipeline.run_pipeline_step"
    ) as patch_run_pipeline_step:
        pipeline.transform()
        patch_run_pipeline_step.assert_called_once_with(
            pipeline_root, pipeline_name, pipeline_steps, pipeline_steps[2]
        )


def test_train_step():
    pipeline_root = "pipeline-root"
    pipeline_name = "sklearn_regression"
    profile = "production"
    pipeline_steps = ["ingestStep", "splitStep", "transformStep", "trainStep", "evaluateStep"]
    pipeline = Pipeline(profile, pipeline_root)
    with patch.object(
        pipeline,
        "resolve_pipeline_steps",
        return_value=(pipeline_name, pipeline_steps),
    ), patch(
        "mlflow.pipelines.regression.v1.pipeline.run_pipeline_step"
    ) as patch_run_pipeline_step:
        pipeline.train()
        patch_run_pipeline_step.assert_called_once_with(
            pipeline_root, pipeline_name, pipeline_steps, pipeline_steps[3]
        )


def test_evaluate_step():
    pipeline_root = "pipeline-root"
    pipeline_name = "sklearn_regression"
    profile = "production"
    pipeline_steps = ["ingestStep", "splitStep", "transformStep", "trainStep", "evaluateStep"]
    pipeline = Pipeline(profile, pipeline_root)
    with patch.object(
        pipeline,
        "resolve_pipeline_steps",
        return_value=(pipeline_name, pipeline_steps),
    ), patch(
        "mlflow.pipelines.regression.v1.pipeline.run_pipeline_step"
    ) as patch_run_pipeline_step:
        pipeline.evaluate()
        patch_run_pipeline_step.assert_called_once_with(
            pipeline_root, pipeline_name, pipeline_steps, pipeline_steps[4]
        )
