from examples.pipelines.sklearn_regression.steps.train import train_fn
from sklearn.utils.estimator_checks import check_estimator


def test_user_train_returns_object_with_correct_spec():
    regressor = train_fn()
    assert callable(getattr(regressor, "fit", None))
    assert callable(getattr(regressor, "predict", None))


def test_user_train_passes_check_estimator():
    regressor = train_fn()
    check_estimator(regressor)
