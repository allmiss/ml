import pandas as pd
from sklearn.pipeline import Pipeline

from src.models import ChurnPrediction, FeatureVectorChurn


def predict_churn(pipeline: Pipeline, features: list[FeatureVectorChurn]) -> list[ChurnPrediction]:
    if not features:
        return []

    X = pd.DataFrame([item.model_dump() for item in features])
    classes = pipeline.classes_
    labels = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)

    return [
        ChurnPrediction(
            churn=int(label),
            probabilities={int(cls): round(float(p), 4) for cls, p in zip(classes, probs)},
        )
        for label, probs in zip(labels, probabilities)
    ]
