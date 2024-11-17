import os
from typing import Dict, List, Tuple
import yaml
import pandas as pd
from sklearn import metrics
from qmlab.preprocessing import parse_biomed_data_to_ndarray
from sklearn.model_selection import StratifiedKFold
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
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    result: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for data in datasets:
        result[data] = {}
        result_data: Dict[str, List[float]] = {"train_acc": [], "test_acc": []}
        for kernel in kernels:
            X, y = parse_biomed_data_to_ndarray(data, return_X_y=True)

            skf = StratifiedKFold(
                n_splits=num_splits, shuffle=True, random_state=random_state
            )
            clf = SVC(kernel=kernel, random_state=random_state)

            for train_idx, test_idx in skf.split(X, y):
                X_train = X[train_idx, :]
                X_test = X[test_idx, :]
                y_train = y[train_idx]
                y_test = y[test_idx]

                clf.fit(X_train, y_train)

                train_pred = clf.predict(X_train)
                test_pred = clf.predict(X_test)

                train_acc = metrics.balanced_accuracy_score(y_train, train_pred)
                test_acc = metrics.balanced_accuracy_score(y_test, test_pred)

                result_data["train_acc"].append(train_acc)
                result_data["test_acc"].append(test_acc)

            result[data][kernel] = result_data

    return result


def create_dataframe_from_results(
    results: Dict[str, Dict[str, Dict[str, List[float]]]]
) -> pd.DataFrame:
    tabular_data = []

    for dataset_name, kernels_data in results.items():
        for kernel, acc_data in kernels_data.items():
            # Average accuracy values across the num of folds
            avg_train_acc = sum(acc_data["train_acc"]) / len(acc_data["train_acc"])
            avg_test_acc = sum(acc_data["test_acc"]) / len(acc_data["test_acc"])

            # Append a row to table_data
            tabular_data.append(
                {
                    "dataset": dataset_name,
                    "kernel": kernel,
                    "train_acc": avg_train_acc,
                    "test_acc": avg_test_acc,
                }
            )

    df = pd.DataFrame(tabular_data)
    return df


if __name__ == "__main__":
    results = run_svm_cross_validation(datasets, kernels, num_splits=5)
    data = create_dataframe_from_results(results)

    results_filename = "Classical_SVC_results.csv"
    path_out = os.path.join(out_dir, results_filename)
    df = pd.DataFrame.from_dict(data)
    df.to_csv(path_out)
