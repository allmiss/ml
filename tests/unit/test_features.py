from src.models import FeatureVectorChurn
from src.preprocessing import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET, features_to_frame
from tests.synthetic import make_row


def test_features_match_request_schema() -> None:
    assert set(FEATURES) == set(FeatureVectorChurn.model_fields)
    assert not set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)


def test_features_to_frame_order() -> None:
    features = [FeatureVectorChurn(**{k: v for k, v in make_row().items() if k != TARGET})]
    frame = features_to_frame(features)
    assert list(frame.columns) == FEATURES
    assert len(frame) == 1
