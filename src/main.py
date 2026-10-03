from collections import Counter

from fastapi import FastAPI, Query

from src.dataset import load_dataset
from src.models import DatasetRowChurn, FeatureVectorChurn

app = FastAPI()

dataset = load_dataset()


@app.get("/")
def root():
    return {"message": "ml churn service is running"}


@app.post("/predict")
def predict(features: FeatureVectorChurn) -> FeatureVectorChurn:
    return features


@app.get("/dataset/preview")
def dataset_preview(n: int = Query(5, ge=1)) -> list[DatasetRowChurn]:
    return dataset[:n]


@app.get("/dataset/info")
def dataset_info():
    return {
        "rows": len(dataset),
        "columns": len(DatasetRowChurn.model_fields),
        "features": list(FeatureVectorChurn.model_fields),
        "churn_distribution": dict(sorted(Counter(row.churn for row in dataset).items())),
    }
