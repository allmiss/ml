from sklearn.pipeline import Pipeline

from src.exceptions import FeatureMismatchError
from src.models import ChurnPrediction, FeatureVectorChurn
from src.preprocessing import features_to_frame


def predict_churn(pipeline: Pipeline, features: list[FeatureVectorChurn]) -> list[ChurnPrediction]:
    if not features:
        return []

    X = features_to_frame(features)
    trained_features = list(pipeline.feature_names_in_)
    if list(X.columns) != trained_features:
        raise FeatureMismatchError(
            f"Модель обучена на признаках {trained_features}, а сервис передаёт {list(X.columns)}. "
            "Переобучите модель через POST /model/train"
        )

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
