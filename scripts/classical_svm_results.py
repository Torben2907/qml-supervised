import os
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from sklearn.svm import SVC
from qmlab.utils import run_cv
from qmlab.preprocessing import (
    parse_biomed_data_to_ndarray,
    subsample_features,
    scale_to_specified_interval,
)

random_state = 42

kernels = ("rbf", "poly", "sigmoid")

res_dir = os.path.join(os.path.dirname(__file__), "../res/")

path_to_data = os.path.join(os.path.dirname(__file__), "../data_names.yaml")
with open(path_to_data) as file:
    datasets: list[str] = yaml.safe_load(file)


def compute_svm_results(
    dataset: str,
    kernels: Tuple[str, ...],
    num_splits: int = 5,
    random_state: int = 42,
    num_features_to_subsample: int = 10,
) -> pd.DataFrame:
    results_summary = []
    X, y, feature_names = parse_biomed_data_to_ndarray(dataset, return_X_y=True)
    X = scale_to_specified_interval(X, interval=(-np.pi / 2, np.pi / 2))
    for kernel in tqdm(kernels, desc=f"Kernels ({dataset})"):
        entry = {"Dataset": dataset, "Kernel": kernel}
        subsampled_results = subsample_features(
            X, feature_names, num_features_to_subsample
        )
        svm = SVC(kernel=kernel, probability=True, random_state=random_state)
        for X_sub, feature_names_sub in subsampled_results:
            group_name = str(feature_names_sub)
            results = run_cv(
                svm, X_sub, y, num_splits=num_splits, random_state=random_state
            )
            acc, pr, auc, mcc = tuple(results.values())
            entry[group_name] = f"{acc:.9f}, {pr:.9f}, {auc:.9f}, {mcc:.9f}"
        results_summary.append(entry)
        del svm
    return pd.DataFrame(results_summary)


datasets = ["haberman_new"]


if __name__ == "__main__":
    for data in tqdm(datasets, desc="Datasets"):
        df = compute_svm_results(data, kernels)
        res_name = f"SVM_{data}_results.csv"
        path_out = os.path.join(res_dir, res_name)
        df.to_csv(path_out, index=False)
        print(f"Results saved to {path_out}")
