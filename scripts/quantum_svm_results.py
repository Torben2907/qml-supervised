import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import yaml
import time
from tqdm import tqdm
from qmlab.preprocessing import (
    parse_biomed_data_to_ndarray,
    subsample_features,
    scale_to_specified_interval,
)
from qmlab.kernel import FidelityQuantumKernel
from qmlab.kernel import QSVC
from qmlab.utils import run_cv


def compute_qsvm_results(
    dataset: str,
    data_embeddings: Tuple[str, ...],
    times: List[Dict[str, str | float]],
    num_splits: int = 5,
    random_state: int = 42,
    num_features_to_subsample: int = 11,
) -> pd.DataFrame:
    results_summary = []
    X, y, feature_names = parse_biomed_data_to_ndarray(dataset, return_X_y=True)
    X = scale_to_specified_interval(X, interval=(-np.pi / 2, np.pi / 2))
    for embedding in tqdm(data_embeddings, desc=f"Embedding ({dataset})"):
        start_time = time.time()
        time_elapsed = 0.0
        entry = {"Dataset": dataset, "Embedding": embedding}
        subsampling_results = subsample_features(
            X, feature_names, num_features_to_subsample
        )
        qkernel = FidelityQuantumKernel(data_embedding=embedding, jit=True)
        qsvm = QSVC(quantum_kernel=qkernel, random_state=random_state)
        for X_sub, feature_names_sub in subsampling_results:
            group_name = str(feature_names_sub)
            results = run_cv(
                qsvm, X_sub, y, num_splits=num_splits, random_state=random_state
            )
            mcc = results["mcc"]
            mean = mcc["mean"]
            CI = mcc["CI"]
            if isinstance(CI, list):
                rounded_CI = [round(value, 5) for value in CI]
            entry[group_name] = f"{mean:.5f}, CI: {rounded_CI}"
        results_summary.append(entry)
        time_elapsed = time.time() - start_time
        times.append(
            {"Dataset": dataset, "Embedding": embedding, "Times": time_elapsed}
        )
        del (qkernel, qsvm)
    return pd.DataFrame(results_summary), pd.DataFrame(times)


res_dir = os.path.join(os.path.dirname(__file__), "../res/")
os.makedirs(res_dir, exist_ok=True)
path_to_data_names = os.path.join(os.path.dirname(__file__), "../data_names.yaml")
with open(path_to_data_names) as file:
    datasets: list[str] = yaml.safe_load(file)

data_embeddings = ("Angle", "IQP")
times: List = []

if __name__ == "__main__":
    for data in tqdm(datasets, desc="Datasets"):
        results, times_df = compute_qsvm_results(data, data_embeddings, times)
        res_name = f"QSVM_{data}_results.csv"
        path_out = os.path.join(res_dir, res_name)
        results.to_csv(path_out, index=False)
        times_df.to_csv(os.path.join(res_dir, f"QSVM_{data}_times.csv"))
        print(f"Results saved to {path_out}.")
