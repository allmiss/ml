from collections import Counter

import pandas as pd

from src.dataset import ChurnDataset
from src.models import DatasetInfo, DatasetRowChurn, FeatureVectorChurn, SplitInfo
from src.preprocessing import class_distribution, prepare_data, split_data


class DatasetService:
    def __init__(self, dataset: ChurnDataset):
        self.dataset = dataset

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

    def split(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return split_data(*prepare_data(self.dataset.to_frame()))

    def split_info(self) -> SplitInfo:
        X_train, X_test, y_train, y_test = self.split()
        return SplitInfo(
            train_size=len(X_train),
            test_size=len(X_test),
            churn_distribution={"train": class_distribution(y_train), "test": class_distribution(y_test)},
        )
