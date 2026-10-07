from collections import Counter

import pandas as pd
from fastapi import FastAPI, Query

from src.dataset import load_dataset, to_records
from src.models import DatasetRowChurn, FeatureVectorChurn
from src.preprocessing import churn_distribution, prepare_data, split_data

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


@app.get("/dataset/split-info")
def dataset_split_info():
    X, y, _, _ = prepare_data(pd.DataFrame(to_records(dataset)))
    X_train, X_test, y_train, y_test = split_data(X, y)
    return {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "churn_distribution": churn_distribution(y_train, y_test),
    }
