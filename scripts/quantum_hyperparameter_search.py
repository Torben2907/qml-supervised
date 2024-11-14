import os
import pandas as pd
import numpy as np
from typing import List
import yaml
from sklearn import metrics
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit.primitives import Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_algorithms.utils import algorithm_globals
from qmlab.kernel import FidelityQuantumKernel
from qmlab.preprocessing import (
    parse_biomed_data_to_ndarray,
    reduce_feature_dim,
    scale_to_specified_range,
)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

feature_dimension = 2
train_size = 0.7
random_state = 12345
scoring = "balanced_accuracy"

hyper_parameter_settings = {
    "QSVC": {
        "feature_map": [
            ZZFeatureMap(feature_dimension),
            ZFeatureMap(feature_dimension),
        ],
    }
}

dir_out = os.path.join(os.path.dirname(__file__), "../res/")
file_path = os.path.join(os.path.dirname(__file__), "../data_names.yaml")
with open(file_path) as file:
    data_names: list[str] = yaml.safe_load(file)

if __name__ == "__main__":
    for data in data_names:
        for model in hyper_parameter_settings.keys():
            np.random.seed(random_state)
            algorithm_globals.random_seed = random_state

            X, y = parse_biomed_data_to_ndarray(data)
            X = reduce_feature_dim(X, output_dimension=feature_dimension)
            X = scale_to_specified_range(X, range=(-1.0, 1.0))
            num_samples, num_features = X.shape

            if num_samples > 200:
                X = X[:200, :]
                y = y[:200]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_size, random_state=random_state
            )

            feature_maps = hyper_parameter_settings[model]["feature_map"]

            for fm in feature_maps:
                if isinstance(fm, QuantumCircuit):
                    pass
                else:
                    raise ValueError("feature map must be of type quantum circuit!")

                clf = SVC(kernel="precomputed", random_state=random_state)
                quantum_kernel = FidelityQuantumKernel(
                    feature_map=fm,
                    fidelity=ComputeUncompute(sampler=Sampler()),
                )

                kernel_train = quantum_kernel.evaluate_kernel(X_train)
                kernel_test = quantum_kernel.evaluate_kernel(X_test, X_train)

                clf.fit(kernel_train, y_train)

                train_pred = clf.predict(kernel_train)
                test_pred = clf.predict(kernel_test)

                results_with_best_params: dict[str, List] = {
                    "train_acc": [],
                    "test_acc": [],
                }

                train_acc = metrics.balanced_accuracy_score(y_train, train_pred)

                results_with_best_params["train_acc"].append(train_acc)

                test_acc = metrics.balanced_accuracy_score(y_test, test_pred)

                results_with_best_params["test_acc"].append(test_acc)

                results_filename = " ".join([model + "_" + data.upper()])
                path_out = os.path.join(
                    dir_out, results_filename + "_best_hypers_results.csv"
                )

                df = pd.DataFrame.from_dict(results_with_best_params)
                df.to_csv(path_out)
