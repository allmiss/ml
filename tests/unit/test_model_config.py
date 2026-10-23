import pytest
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.core.exceptions import InvalidTrainingConfigError
from src.ml.training import DEFAULT_HYPERPARAMETERS, create_model, create_pipeline, resolve_config
from src.schemas.model import TrainingConfigChurn

LOGREG = TrainingConfigChurn(model_type="logreg")
FOREST = TrainingConfigChurn(model_type="random_forest", hyperparameters={"n_estimators": 50, "max_depth": 6})


def test_resolve_config_adds_defaults() -> None:
    config = resolve_config(TrainingConfigChurn(model_type="logreg", hyperparameters={"C": 0.5}))
    assert config.hyperparameters == {**DEFAULT_HYPERPARAMETERS["logreg"], "C": 0.5}


def test_resolve_config_user_value_overrides_default() -> None:
    config = resolve_config(TrainingConfigChurn(model_type="logreg", hyperparameters={"max_iter": 50}))
    assert config.hyperparameters["max_iter"] == 50


def test_resolve_config_unknown_model_type() -> None:
    with pytest.raises(InvalidTrainingConfigError) as exc:
        resolve_config(TrainingConfigChurn(model_type="xgboost"))
    assert exc.value.details["supported"] == ["logreg", "random_forest"]


@pytest.mark.parametrize(
    ("config", "model_class"),
    [(LOGREG, LogisticRegression), (FOREST, RandomForestClassifier)],
    ids=["logreg", "random_forest"],
)
def test_create_model_type(config: TrainingConfigChurn, model_class: type[ClassifierMixin]) -> None:
    assert isinstance(create_model(config), model_class)


def test_create_model_passes_hyperparameters() -> None:
    model = create_model(FOREST)
    assert model.n_estimators == 50
    assert model.max_depth == 6


def test_create_model_unknown_hyperparameter() -> None:
    with pytest.raises(InvalidTrainingConfigError, match="Неверные гиперпараметры"):
        create_model(TrainingConfigChurn(model_type="logreg", hyperparameters={"foo": 1}))


def test_create_pipeline_steps() -> None:
    pipeline = create_pipeline(LOGREG)
    assert list(pipeline.named_steps) == ["preprocessor", "model"]
