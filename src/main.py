import logging
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse

from src.config import DATASET_PATH, MODEL_PATH
from src.dataset import ChurnDataset, DatasetUnavailableError
from src.models import DatasetInfo, DatasetRowChurn, FeatureVectorChurn, ModelStatus, SplitInfo, TrainMetrics
from src.preprocessing import class_distribution, prepare_data, split_data
from src.storage import load_churn_model, save_churn_model
from src.training import TrainedModel, evaluate_model, train_churn_model

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.dataset = ChurnDataset(DATASET_PATH)
    app.state.model = None
    try:
        app.state.model = load_churn_model(MODEL_PATH)
        logger.info("Модель загружена из %s", MODEL_PATH)
    except FileNotFoundError:
        logger.info("Сохранённой модели нет, обучите её через POST /model/train")
    except Exception:
        logger.exception("Не удалось загрузить модель из %s", MODEL_PATH)
    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(DatasetUnavailableError)
async def dataset_unavailable_handler(request: Request, exc: DatasetUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


def get_dataset(request: Request) -> ChurnDataset:
    return request.app.state.dataset


DatasetDep = Annotated[ChurnDataset, Depends(get_dataset)]


def get_model(request: Request) -> TrainedModel | None:
    return request.app.state.model


ModelDep = Annotated[TrainedModel | None, Depends(get_model)]


@app.get("/")
def root():
    return {"message": "ml churn service is running"}


@app.post("/predict")
def predict(features: FeatureVectorChurn) -> FeatureVectorChurn:
    return features


@app.get("/dataset/preview")
def dataset_preview(dataset: DatasetDep, n: int = Query(5, ge=1)) -> list[DatasetRowChurn]:
    return dataset.rows[:n]


@app.get("/dataset/info")
def dataset_info(dataset: DatasetDep) -> DatasetInfo:
    return DatasetInfo(
        rows=len(dataset.rows),
        columns=len(DatasetRowChurn.model_fields),
        features=list(FeatureVectorChurn.model_fields),
        churn_distribution=dict(sorted(Counter(row.churn for row in dataset.rows).items())),
    )


@app.get("/dataset/split-info")
def dataset_split_info(dataset: DatasetDep) -> SplitInfo:
    X_train, X_test, y_train, y_test = split_data(*prepare_data(dataset.to_frame()))
    return SplitInfo(
        train_size=len(X_train),
        test_size=len(X_test),
        churn_distribution={"train": class_distribution(y_train), "test": class_distribution(y_test)},
    )


@app.post("/model/train")
def model_train(request: Request, dataset: DatasetDep) -> TrainMetrics:
    X_train, X_test, y_train, y_test = split_data(*prepare_data(dataset.to_frame()))
    pipeline = train_churn_model(X_train, y_train)
    model = TrainedModel(
        pipeline=pipeline,
        trained_at=datetime.now().astimezone(),
        metrics=evaluate_model(pipeline, X_test, y_test),
    )
    save_churn_model(model)
    request.app.state.model = model
    return model.metrics


@app.get("/model/status")
def model_status(model: ModelDep) -> ModelStatus:
    if model is None:
        return ModelStatus(trained=False)
    return ModelStatus(trained=True, trained_at=model.trained_at, metrics=model.metrics)
