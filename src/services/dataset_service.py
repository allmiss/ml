from collections import Counter

from src.core.exceptions import ChurnServiceError
from src.ml.dataset import ChurnDataset
from src.ml.preprocessing import DataSplit, class_distribution, prepare_data, split_data
from src.schemas.churn import DatasetRowChurn, FeatureVectorChurn
from src.schemas.dataset import DatasetInfo, SplitInfo
from src.schemas.health import DatasetHealth


class DatasetService:
    def __init__(self, dataset: ChurnDataset):
        self.dataset = dataset

    def health(self) -> DatasetHealth:
        # датасет грузится при первом обращении и дальше берётся из памяти,
        # поэтому повторные проверки файл не перечитывают
        try:
            return DatasetHealth(loaded=True, rows=len(self.dataset.rows))
        except ChurnServiceError as e:
            return DatasetHealth(loaded=False, error=e.message)

    def preview(self, n: int) -> list[DatasetRowChurn]:
        return self.dataset.rows[:n]

    def info(self) -> DatasetInfo:
        rows = self.dataset.rows
        return DatasetInfo(
            rows=len(rows),
            columns=len(DatasetRowChurn.model_fields),
            features=list(FeatureVectorChurn.model_fields),
            churn_distribution=dict(sorted(Counter(row.churn for row in rows).items())),
        )

    def split(self) -> DataSplit:
        return split_data(*prepare_data(self.dataset.to_frame()))

    def split_info(self) -> SplitInfo:
        X_train, X_test, y_train, y_test = self.split()
        return SplitInfo(
            train_size=len(X_train),
            test_size=len(X_test),
            churn_distribution={"train": class_distribution(y_train), "test": class_distribution(y_test)},
        )
