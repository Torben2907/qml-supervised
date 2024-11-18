import os
from typing import Dict, List, Tuple
import yaml
import pandas as pd
from sklearn import metrics
from qmlab.preprocessing import parse_biomed_data_to_ndarray
from sklearn.model_selection import ShuffleSplit
from qmlab.kernel import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap
from sklearn.svm import SVC

random_state = 42

kernels = ("rbf", "poly", "linear", "sigmoid")

out_dir = os.path.join(os.path.dirname(__file__), "../res/")

path_to_data = os.path.join(os.path.dirname(__file__), "../data_names.yaml")
with open(path_to_data) as file:
    datasets: list[str] = yaml.safe_load(file)


def run_svm_cross_validation(
    datasets: List[str],
    kernels: Tuple[str, ...],
    num_splits: int = 5,
    random_state: int = 42,
    test_size: float = 0.3,
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    results: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for data in datasets:
        results[data] = {}
        for kernel in kernels:
            X, y, feature_names = parse_biomed_data_to_ndarray(data, return_X_y=True)
            num_samples, num_features = X.shape
            result_data: Dict[str, List[float]] = {"train_acc": [], "test_acc": []}

            ss = ShuffleSplit(
                n_splits=num_splits, test_size=test_size, random_state=random_state
            )
            svc = SVC(kernel=kernel, random_state=random_state)

            for train_idx, test_idx in ss.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                svc.fit(X_train, y_train)

                train_pred = svc.predict(X_train)
                test_pred = svc.predict(X_test)

                train_acc = metrics.accuracy_score(y_train, train_pred)
                test_acc = metrics.accuracy_score(y_test, test_pred)

                result_data["train_acc"].append(train_acc)
                result_data["test_acc"].append(test_acc)

            results[data][kernel] = result_data

    return results


def create_dataframe_from_results(
    results: Dict[str, Dict[str, Dict[str, List[float]]]]
) -> pd.DataFrame:
    tabular_data = []
    for dataset_name, kernels_data in results.items():
        for kernel, acc_data in kernels_data.items():
            avg_train_acc = sum(acc_data["train_acc"]) / len(acc_data["train_acc"])
            avg_test_acc = sum(acc_data["test_acc"]) / len(acc_data["test_acc"])

            tabular_data.append(
                {
                    "dataset": dataset_name,
                    "kernel": kernel,
                    "avg_train_acc": avg_train_acc,
                    "avg_test_acc": avg_test_acc,
                }
            )

    df = pd.DataFrame(tabular_data)
    return df


if __name__ == "__main__":
    results = run_svm_cross_validation(
        datasets, kernels, num_splits=10, random_state=random_state, test_size=0.3
    )
    print(type(results.items()))
    data = create_dataframe_from_results(results)
    results_filename = "Classical_SVC_results.csv"
    path_out = os.path.join(out_dir, results_filename)
    df = pd.DataFrame.from_dict(data)
    df.to_csv(path_out)
