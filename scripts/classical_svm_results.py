import os
import yaml
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple
from sklearn.svm import SVC
from qmlab.utils import run_cross_validation
from qmlab.preprocessing import parse_biomed_data_to_ndarray, subsample_features

random_state = 42

kernels = ("rbf", "poly", "linear", "sigmoid")

out_dir = os.path.join(os.path.dirname(__file__), "../res/")

path_to_data = os.path.join(os.path.dirname(__file__), "../data_names.yaml")
with open(path_to_data) as file:
    datasets: list[str] = yaml.safe_load(file)


def compute_svm_results(
    datasets: List[str],
    kernels: Tuple[str, ...],
    num_splits: int = 5,
    random_state: int = 42,
    num_features_to_subsample: int = 4,
) -> pd.DataFrame:
    results_summary = []
    for dataset in tqdm(datasets, desc="Datasets"):
        for kernel in tqdm(kernels, desc="Kernels"):
            entry = {"Dataset": dataset, "Kernel": kernel}
            X, y, feature_names = parse_biomed_data_to_ndarray(dataset, return_X_y=True)
            subsampled_results = subsample_features(
                X, feature_names, num_features_to_subsample
            )
            svm = SVC(kernel=kernel, probability=True, random_state=random_state)
            for X_sub, feature_names_sub in subsampled_results:
                group_name = str(feature_names_sub)
                results = run_cross_validation(
                    svm, X_sub, y, num_splits=num_splits, random_state=random_state
                )
                acc, f1, mcc, auc = tuple(results.values())
                entry[group_name] = f"{acc:.5f}, {mcc:.5f}, {f1:.5f}, {auc:.5f}"
            results_summary.append(entry)
    return pd.DataFrame(results_summary)


if __name__ == "__main__":
    df = compute_svm_results(
        datasets,
        kernels,
        num_splits=10,
        random_state=random_state,
        num_features_to_subsample=3,
    )
    results_filename = "Classical_SVC_results.csv"
    path_out = os.path.join(out_dir, results_filename)
    df.to_csv(path_out)
