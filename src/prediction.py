from sklearn.pipeline import Pipeline

from src.exceptions import FeatureMismatchError, PredictionError
from src.models import ChurnPrediction, FeatureVectorChurn
from src.preprocessing import features_to_frame


def predict_churn(pipeline: Pipeline, features: list[FeatureVectorChurn]) -> list[ChurnPrediction]:
    if not features:
        return []

    X = features_to_frame(features)
    trained_features = list(pipeline.feature_names_in_)
    if list(X.columns) != trained_features:
        raise FeatureMismatchError(
            "Признаки сервиса не совпадают с признаками модели, переобучите её через POST /model/train",
            details={"expected": trained_features, "received": list(X.columns)},
        )

    try:
        classes = pipeline.classes_
        labels = pipeline.predict(X)
        probabilities = pipeline.predict_proba(X)
    except (ValueError, TypeError, AttributeError) as e:
        raise PredictionError(
            "Модель не смогла посчитать предсказание",
            details={"clients": len(features), "reason": str(e)},
        ) from None

    return [
        ChurnPrediction(
            churn=int(label),
            probabilities={int(cls): round(float(p), 4) for cls, p in zip(classes, probs)},
        )
        for label, probs in zip(labels, probabilities)
    ]
