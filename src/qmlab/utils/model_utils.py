from typing import Dict, List  # noqa: F401
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin


def run_cross_validation(
    clf: BaseEstimator | ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    num_splits: int = 10,
    random_state: int = 42,
) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=num_splits, random_state=random_state, shuffle=True)
    accuracies = []
    # precisions = []
    # recalls = []
    # f1_scores = []
    # auc_scores = []
    mccs = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        # y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = metrics.accuracy_score(y_test, y_pred)
        # precision = metrics.precision_score(y_test, y_pred)
        # recall = metrics.recall_score(y_test, y_pred)
        # f1 = metrics.f1_score(y_test, y_pred)
        mcc = metrics.matthews_corrcoef(y_test, y_pred)
        # auc = metrics.roc_auc_score(y_test, y_pred_proba)

        accuracies.append(accuracy)
        # precisions.append(precision)
        # recalls.append(recall)
        # f1_scores.append(f1)
        mccs.append(mcc)
        # auc_scores.append(auc)

    results = {
        "accuracy": np.mean(accuracies).item(),
        # "recall": np.mean(recalls).item(),
        # "f1": np.mean(f1_scores).item(),
        # "auc": np.mean(auc_scores).item(),
        "mcc": np.mean(mccs).item(),
    }

    return results
