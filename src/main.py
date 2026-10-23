from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.config import DATASET_PATH, HISTORY_PATH, MODEL_PATH
from src.dataset import ChurnDataset
from src.handlers.exception_handlers import register_exception_handlers
from src.history import TrainingHistory
from src.logging_config import setup_logging
from src.routes import dataset, health, model, prediction, root
from src.services.dataset_service import DatasetService
from src.services.model_service import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    dataset_service = DatasetService(ChurnDataset(DATASET_PATH))
    model_service = ModelService(MODEL_PATH, dataset_service, TrainingHistory(HISTORY_PATH))
    model_service.load()

    app.state.dataset_service = dataset_service
    app.state.model_service = model_service
    yield


setup_logging()

app = FastAPI(lifespan=lifespan)
register_exception_handlers(app)

app.include_router(root.router)
app.include_router(health.router)
app.include_router(dataset.router)
app.include_router(model.router)
app.include_router(prediction.router)
