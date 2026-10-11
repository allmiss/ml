import logging
from datetime import datetime
from pathlib import Path

from src.exceptions import ModelNotTrainedError
from src.models import FeatureVectorChurn, ModelStatus, PredictionResponseChurn, TrainMetrics
from src.prediction import predict_churn
from src.services.dataset_service import DatasetService
from src.storage import load_churn_model, save_churn_model
from src.training import TrainedModel, evaluate_model, train_churn_model

logger = logging.getLogger("uvicorn.error")


class ModelService:
    def __init__(self, model_path: Path, dataset_service: DatasetService):
        self.model_path = model_path
        self.dataset_service = dataset_service
        self.model: TrainedModel | None = None

    def load(self) -> None:
        try:
            self.model = load_churn_model(self.model_path)
            logger.info("Модель загружена из %s", self.model_path)
        except FileNotFoundError:
            logger.info("Сохранённой модели нет, обучите её через POST /model/train")
        except Exception:
            logger.exception("Не удалось загрузить модель из %s", self.model_path)

    def train(self) -> TrainMetrics:
        X_train, X_test, y_train, y_test = self.dataset_service.split()
        pipeline = train_churn_model(X_train, y_train)
        model = TrainedModel(
            pipeline=pipeline,
            trained_at=datetime.now().astimezone(),
            metrics=evaluate_model(pipeline, X_test, y_test),
        )
        save_churn_model(model, self.model_path)
        self.model = model
        return model.metrics

    def status(self) -> ModelStatus:
        if self.model is None:
            return ModelStatus(trained=False)
        return ModelStatus(trained=True, trained_at=self.model.trained_at, metrics=self.model.metrics)

    def predict(self, features: list[FeatureVectorChurn]) -> PredictionResponseChurn:
        if self.model is None:
            raise ModelNotTrainedError("Модель не обучена, обучите её через POST /model/train")
        return PredictionResponseChurn(predictions=predict_churn(self.model.pipeline, features))
