import os
import yaml
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from sklearn.svm import SVC
from qmlab.utils import run_cv_roc_analysis
from qmlab.preprocessing import (
    parse_biomed_data_to_ndarray,
    subsample_features,
    scale_to_specified_interval,
)

random_state = 42

kernels = ("rbf", "poly", "sigmoid")

res_dir = os.path.join(os.path.dirname(__file__), "../roc-analysis/")
os.makedirs(res_dir, exist_ok=True)


path_to_data = os.path.join(os.path.dirname(__file__), "../data_names.yaml")
with open(path_to_data) as file:
    datasets: list[str] = yaml.safe_load(file)


def svm_roc_analysis(
    dataset: str,
    kernels: Tuple[str, ...],
    output_dir: str,
    num_splits: int = 5,
    random_state: int = 42,
    num_features_to_subsample: int = 10,
) -> List:
    X, y, feature_names = parse_biomed_data_to_ndarray(dataset, return_X_y=True)
    X = scale_to_specified_interval(X, interval=(-np.pi / 2, np.pi / 2))
    all_plot_files = []
    for kernel in tqdm(kernels, desc=f"Kernels ({dataset})"):
        subsampled_results = subsample_features(
            X, feature_names, num_features_to_subsample
        )
        svm = SVC(kernel=kernel, probability=True, random_state=random_state)
        for X_sub, feature_names_sub in subsampled_results:
            plot_files = run_cv_roc_analysis(
                svm,
                X_sub,
                feature_names_sub,
                y,
                num_splits=num_splits,
                random_state=random_state,
                output_dir=output_dir,
                kernel_name=kernel,
                dataset_name=dataset,
            )
            all_plot_files.extend(plot_files)
        del svm
    print(f"All plots for {dataset} saved in {output_dir}")
    return all_plot_files


if __name__ == "__main__":
    for data in tqdm(datasets, desc="Datasets"):
        df = svm_roc_analysis(data, kernels, res_dir)