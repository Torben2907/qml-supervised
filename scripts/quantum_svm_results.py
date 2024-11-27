import os
import numpy as np
import pandas as pd
from typing import Tuple
import yaml
from tqdm import tqdm
from qmlab.preprocessing import (
    parse_biomed_data_to_ndarray,
    subsample_features,
    scale_to_specified_interval,
)
from qmlab.kernel import FidelityQuantumKernel
from qmlab.kernel import QSVC
from qmlab.utils import run_cross_validation


def compute_qsvm_results_for_dataset(
    dataset: str,
    data_embeddings: Tuple[str, ...],
    num_splits: int = 5,
    random_state: int = 42,
    num_features_to_subsample: int = 9,
) -> pd.DataFrame:
    results_summary = []
    X, y, feature_names = parse_biomed_data_to_ndarray(dataset, return_X_y=True)
    X = scale_to_specified_interval(X, interval=(-np.pi / 2, np.pi / 2))
    for embedding in tqdm(data_embeddings, desc=f"Embedding ({dataset})"):
        entry = {"Dataset": dataset, "Kernel": embedding}
        subsampling_results = subsample_features(
            X, feature_names, num_features_to_subsample
        )
        qkernel = FidelityQuantumKernel(data_embedding=embedding, jit=True)
        qsvm = QSVC(quantum_kernel=qkernel, random_state=random_state)
        for X_sub, feature_names_sub in subsampling_results:
            group_name = str(feature_names_sub)
            results = run_cross_validation(qsvm, X_sub, y, num_splits, random_state)
            acc, f1, auc, mcc = tuple(results.values())
            entry[group_name] = f"{acc:.9f}, {f1:.9f}, {auc:.9f}, {mcc:.9f}"
        results_summary.append(entry)
        del (qkernel, qsvm)
    return pd.DataFrame(results_summary)


res_dir = os.path.join(os.path.dirname(__file__), "../res/")
path_to_data_names = os.path.join(os.path.dirname(__file__), "../data_names.yaml")
with open(path_to_data_names) as file:
    datasets: list[str] = yaml.safe_load(file)

datasets = ["nafld_new"]
data_embeddings = ("IQP", "Angle")

if __name__ == "__main__":
    for data in tqdm(datasets, desc="Datasets"):
        df = compute_qsvm_results_for_dataset(data, data_embeddings)
        res_name = f"QSVM_{data}_results.csv"
        path_out = os.path.join(res_dir, res_name)
        df.to_csv(path_out, index=False)
        print(f"Results saved to {path_out}")
