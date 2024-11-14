import os
from typing import Dict, List, Tuple
import yaml
import pandas as pd
from sklearn import metrics
from qmlab.preprocessing import parse_biomed_data_to_ndarray
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

random_state = 42

out_dir = os.path.join(os.path.dirname(__file__), "../res/")

path_to_data = os.path.join(os.path.dirname(__file__), "../data_names.yaml")
with open(path_to_data) as file:
    datasets: list[str] = yaml.safe_load(file)

kernel_vals = ("rbf", "poly", "linear")


def run_svc_cross_validation(
    datasets: List[str], kernels: Tuple[str, ...]
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    result: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for data in datasets:
        result[data] = {}
        result_data: Dict[str, List[float]] = {"train_acc": [], "test_acc": []}
        for kernel in kernel_vals:
            X, y = parse_biomed_data_to_ndarray(data)

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
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

            avg_train_acc = sum(result_data["train_acc"]) / len(
                result_data["train_acc"]
            )
            avg_test_acc = sum(result_data["test_acc"]) / len(result_data["test_acc"])
            result[data][kernel] = result_data

    return result


def create_dataframe_from_results(
    results: Dict[str, Dict[str, Dict[str, List[float]]]]
) -> pd.DataFrame:
    tabular_data = []

    for dataset_name, kernels_data in results.items():
        for kernel, acc_data in kernels_data.items():
            # Average accuracy values across the 5 folds
            avg_train_acc = sum(acc_data["train_acc"]) / len(acc_data["train_acc"])
            avg_test_acc = sum(acc_data["test_acc"]) / len(acc_data["test_acc"])

            # Append a row to table_data
            tabular_data.append(
                {
                    "Datasets": dataset_name,
                    "Kernel": kernel,
                    "train_acc": avg_train_acc,
                    "test_acc": avg_test_acc,
                }
            )

    df = pd.DataFrame(tabular_data)
    return df


if __name__ == "__main__":
    results = run_svc_cross_validation(datasets, kernel_vals)
    data = create_dataframe_from_results(results)
    print(data)


# results_filename = "Classical_SVC_results.csv"
# path_out = os.path.join(out_dir, results_filename)

# df = pd.DataFrame.from_dict(result)
# df.to_csv(path_out)
