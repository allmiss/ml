import pytest

from src.core.exceptions import InvalidTrainingConfigError
from src.ml.preprocessing import DataSplit
from src.ml.training import evaluate_model, train_churn_model
from src.schemas.model import TrainingConfigChurn

LOGREG = TrainingConfigChurn(model_type="logreg")
FOREST = TrainingConfigChurn(model_type="random_forest", hyperparameters={"n_estimators": 50, "max_depth": 6})
CONFIGS = pytest.mark.parametrize("config", [LOGREG, FOREST], ids=["logreg", "random_forest"])


@CONFIGS
def test_train_and_evaluate(churn_split: DataSplit, config: TrainingConfigChurn) -> None:
    X_train, X_test, y_train, y_test = churn_split
    pipeline = train_churn_model(X_train, y_train, config)

    assert list(pipeline.feature_names_in_) == list(X_train.columns)
    assert set(pipeline.classes_) == {0, 1}

    metrics = evaluate_model(pipeline, X_test, y_test)
    for value in (metrics.accuracy, metrics.f1, metrics.roc_auc):
        assert 0 <= value <= 1
    # модель должна ранжировать клиентов лучше случайного угадывания
    assert metrics.roc_auc > 0.5


@CONFIGS
def test_training_is_reproducible(churn_split: DataSplit, config: TrainingConfigChurn) -> None:
    X_train, X_test, y_train, y_test = churn_split
    first = evaluate_model(train_churn_model(X_train, y_train, config), X_test, y_test)
    second = evaluate_model(train_churn_model(X_train, y_train, config), X_test, y_test)
    assert first == second


def test_class_weight_balanced_improves_f1(churn_split: DataSplit) -> None:
    X_train, X_test, y_train, y_test = churn_split
    plain = evaluate_model(train_churn_model(X_train, y_train, LOGREG), X_test, y_test)
    balanced_config = TrainingConfigChurn(model_type="logreg", hyperparameters={"class_weight": "balanced"})
    balanced = evaluate_model(train_churn_model(X_train, y_train, balanced_config), X_test, y_test)
    assert balanced.f1 > plain.f1


def test_invalid_hyperparameter_value(churn_split: DataSplit) -> None:
    X_train, _, y_train, _ = churn_split
    config = TrainingConfigChurn(model_type="logreg", hyperparameters={"C": -1})
    with pytest.raises(InvalidTrainingConfigError, match="Не удалось обучить"):
        train_churn_model(X_train, y_train, config)
