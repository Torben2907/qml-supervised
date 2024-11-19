import os
import yaml
from tqdm import tqdm
import numpy as np
from qmlab.preprocessing import (
    parse_biomed_data_to_ndarray,
    subsample_features,
    scale_to_specified_interval,
)
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
from sklearn.svm import SVC
from qmlab.kernel.iqp_kernel import IQPKernelClassifier


out_dir = os.path.join(os.path.dirname(__file__), "../res/")
path_to_data = os.path.join(os.path.dirname(__file__), "../data_names.yaml")
with open(path_to_data) as file:
    datasets: list[str] = yaml.safe_load(file)


if __name__ == "__main__":
    X, y, feature_names = parse_biomed_data_to_ndarray("sobar_new", return_X_y=True)
    X = scale_to_specified_interval(X, interval=(-np.pi, np.pi))
    subsamples = subsample_features(X, feature_names, num_features_to_subsample=3)

    avg_classical_scores = []
    avg_quantum_scores = []
    for X_sub, subsampled_features in tqdm(subsamples):
        classical_scores = []
        quantum_scores = []
        rs = ShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
        qsvc = IQPKernelClassifier(jit=True)
        svc = SVC(kernel="linear", random_state=42)

        for train_idx, test_idx in rs.split(X, y):
            X_train, X_test = X_sub[train_idx], X_sub[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            svc.fit(X_train, y_train)
            qsvc.fit(X_train, y_train)

            y_pred = svc.predict(X_test)
            classical_scores.append(metrics.accuracy_score(y_test, y_pred))

            y_pred = qsvc.predict(X_test)
            quantum_scores.append(metrics.accuracy_score(y_test, y_pred))

        avg_classical_score = sum(classical_scores) / len(classical_scores)
        avg_classical_scores.append(avg_classical_score)

        avg_quantum_score = sum(quantum_scores) / len(quantum_scores)
        avg_quantum_scores.append(avg_quantum_score)

    print(avg_classical_scores)
    print(avg_quantum_scores)