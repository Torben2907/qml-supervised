from typing import Callable, Dict
import jax.numpy as jnp
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import gen_batches


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


def chunk_vmapped_fn(
    vmapped_fn: Callable[..., jnp.ndarray], start: int, max_vmap: int
) -> Callable[..., jnp.ndarray]:
    """
    Convert a vmapped function to an equivalent function that evaluates in chunks of size
    max_vmap. The behaviour of chunked_fn should be the same as vmapped_fn, but with a
    lower memory cost.

    The input vmapped_fn should have in_axes = (None, None, ..., 0,0,...,0)

    Args:
        vmapped (func): vmapped function with in_axes = (None, None, ..., 0,0,...,0)
        start (int): The index where the first 0 appears in in_axes
        max_vmap (int) The max chunk size with which to evaluate the function

    Returns:
        chunked version of the function
    """

    def chunked_fn(*args):
        batch_len = len(args[start])
        batch_slices = list(gen_batches(batch_len, max_vmap))
        res = [
            vmapped_fn(*args[:start], *[arg[slice] for arg in args[start:]])
            for slice in batch_slices
        ]
        # jnp.concatenate needs to act on arrays with the same shape, so pad the last array if necessary
        if batch_len / max_vmap % 1 != 0.0:
            diff = max_vmap - len(res[-1])
            res[-1] = jnp.pad(
                res[-1], [(0, diff), *[(0, 0)] * (len(res[-1].shape) - 1)]
            )
            return jnp.concatenate(res)[:-diff]
        else:
            return jnp.concatenate(res)

    return chunked_fn
