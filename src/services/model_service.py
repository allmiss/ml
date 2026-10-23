import logging
import time
from datetime import datetime
from pathlib import Path

from src.exceptions import ModelNotTrainedError
from src.history import TrainingHistory, compare_configs
from src.models import (
    FeatureInfo,
    FeatureVectorChurn,
    ModelHealth,
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

logger = logging.getLogger(__name__)


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

        # проверяем конфиг до разбиения данных: в лог об обучении попадают только
        # поддерживаемые модели и итоговые гиперпараметры вместе со значениями по умолчанию
        config = resolve_config(config)
        X_train, X_test, y_train, y_test = self.dataset_service.split()
        logger.info(
            "Обучение %s %s: train %d, test %d",
            config.model_type, config.hyperparameters, len(X_train), len(X_test),
        )
        started = time.perf_counter()
        pipeline = train_churn_model(X_train, y_train, config)
        model = TrainedModel(
            pipeline=pipeline,
            config=config,
            trained_at=datetime.now().astimezone(),
            metrics=evaluate_model(pipeline, X_test, y_test),
        )
        save_churn_model(model, self.model_path)
        self.model = model
        record = self.history.add(
            trained_at=model.trained_at,
            model_type=model.config.model_type,
            hyperparameters=model.config.hyperparameters,
            train_size=len(X_train),
            test_size=len(X_test),
            metrics=model.metrics,
        )
        logger.info(
            "Модель %s обучена за %.2f с и сохранена в %s: accuracy=%s f1=%s roc_auc=%s, запись истории #%d",
            model.config.model_type, time.perf_counter() - started, self.model_path.name,
            model.metrics.accuracy, model.metrics.f1, model.metrics.roc_auc, record.id,
        )
        return model.metrics

    def health(self) -> ModelHealth:
        if self.model is None:
            return ModelHealth(available=False)
        return ModelHealth(available=True, model_type=self.model.config.model_type, trained_at=self.model.trained_at)

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
        started = time.perf_counter()
        predictions = predict_churn(self.model.pipeline, features)
        # значения признаков не логируем: это данные клиентов
        logger.info(
            "Предсказание для %d клиентов за %.1f мс: уйдут %d",
            len(predictions), 1000 * (time.perf_counter() - started), sum(p.churn for p in predictions),
        )
        return PredictionResponseChurn(predictions=predictions)
