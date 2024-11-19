from typing import List
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
from tqdm import tqdm


def run_shuffle_split(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    num_splits: int = 10,
    test_size: float = 0.3,
    random_state: int = 42,
) -> List[float]:
    rs = ShuffleSplit(
        n_splits=num_splits, test_size=test_size, random_state=random_state
    )
    acc_scores = []
    for train_idx, test_idx in tqdm(rs.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = metrics.accuracy_score(y_pred, y_test)
        acc_scores.append(acc)

    return acc_scores
