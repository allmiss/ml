import logging
from datetime import datetime
from pathlib import Path

from src.exceptions import ModelNotTrainedError
from src.history import TrainingHistory, compare_configs
from src.models import (
    FeatureInfo,
    FeatureVectorChurn,
    MetricName,
    ModelComparison,
    ModelMetrics,
    ModelSchema,
    ModelStatus,
    PredictionResponseChurn,
    TrainingConfigChurn,
    TrainMetrics,
)
from src.prediction import predict_churn
from src.preprocessing import FEATURES, NUMERIC_FEATURES, TARGET, fitted_categories
from src.services.dataset_service import DatasetService
from src.storage import load_churn_model, save_churn_model
from src.training import TrainedModel, evaluate_model, resolve_config, train_churn_model

logger = logging.getLogger("uvicorn.error")


class ModelService:
    def __init__(self, model_path: Path, dataset_service: DatasetService, history: TrainingHistory):
        self.model_path = model_path
        self.dataset_service = dataset_service
        self.history = history
        self.model: TrainedModel | None = None

    def load(self) -> None:
        try:
            self.model = load_churn_model(self.model_path)
            logger.info("Модель загружена из %s", self.model_path)
        except FileNotFoundError:
            logger.info("Сохранённой модели нет, обучите её через POST /model/train")
        except Exception:
            logger.exception("Не удалось загрузить модель из %s", self.model_path)

    def train(self, config: TrainingConfigChurn) -> TrainMetrics:
        # Читаем историю заранее: если файл повреждён, ошибка будет до обучения,
        # а не после того, как новая модель уже сохранена без записи в истории
        self.history.records

        X_train, X_test, y_train, y_test = self.dataset_service.split()
        pipeline = train_churn_model(X_train, y_train, config)
        model = TrainedModel(
            pipeline=pipeline,
            config=resolve_config(config),
            trained_at=datetime.now().astimezone(),
            metrics=evaluate_model(pipeline, X_test, y_test),
        )
        save_churn_model(model, self.model_path)
        self.model = model
        self.history.add(
            trained_at=model.trained_at,
            model_type=model.config.model_type,
            hyperparameters=model.config.hyperparameters,
            train_size=len(X_train),
            test_size=len(X_test),
            metrics=model.metrics,
        )
        return model.metrics

    def status(self) -> ModelStatus:
        if self.model is None:
            return ModelStatus(trained=False)
        return ModelStatus(
            trained=True,
            model_type=self.model.config.model_type,
            hyperparameters=self.model.config.hyperparameters,
            trained_at=self.model.trained_at,
            metrics=self.model.metrics,
        )

    def metrics(self, last: int, model_type: str | None = None) -> ModelMetrics:
        latest = self.history.last(1, model_type)
        return ModelMetrics(latest=latest[0] if latest else None, history=self.history.last(last, model_type))

    def compare(self, metric: MetricName, model_type: str | None = None) -> ModelComparison:
        records = [r for r in self.history.records if model_type is None or r.model_type == model_type]
        configs = compare_configs(records, metric)
        return ModelComparison(metric=metric, best=configs[0] if configs else None, configs=configs)

    def schema(self) -> ModelSchema:
        types = {name: prop["type"] for name, prop in FeatureVectorChurn.model_json_schema()["properties"].items()}
        categories = fitted_categories(self.model.pipeline) if self.model is not None else {}
        return ModelSchema(
            features=[
                FeatureInfo(
                    name=name,
                    type=types[name],
                    kind="numeric" if name in NUMERIC_FEATURES else "categorical",
                    allowed_values=categories.get(name),
                )
                for name in FEATURES
            ],
            target=TARGET,
        )

    def predict(self, features: list[FeatureVectorChurn]) -> PredictionResponseChurn:
        if self.model is None:
            raise ModelNotTrainedError("Модель не обучена, обучите её через POST /model/train")
        return PredictionResponseChurn(predictions=predict_churn(self.model.pipeline, features))
