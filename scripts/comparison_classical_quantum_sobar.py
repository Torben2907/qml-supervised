import os
import yaml
from tqdm import tqdm
from qmlab.preprocessing import (
    parse_biomed_data_to_ndarray,
    subsample_features,
    scale_to_specified_interval,
)
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
from sklearn.svm import SVC
from qiskit.circuit.library import ZZFeatureMap
from qmlab.kernel.iqp_kernel import IQPKernelClassifier
from qmlab.kernel import FidelityQuantumKernel

out_dir = os.path.join(os.path.dirname(__file__), "../res/")
path_to_data = os.path.join(os.path.dirname(__file__), "../data_names.yaml")
with open(path_to_data) as file:
    datasets: list[str] = yaml.safe_load(file)


if __name__ == "__main__":
    X, y, feature_names = parse_biomed_data_to_ndarray("haberman_new", return_X_y=True)
    subsamples = subsample_features(X, feature_names, num_features_to_subsample=3)

    avg_classical_scores = []
    avg_quantum_scores = []
    for X_sub, subsampled_features in tqdm(subsamples):
        classical_scores = []
        quantum_scores = []
        rs = ShuffleSplit(n_splits=5, test_size=0.3, random_state=42)

        qkernel = FidelityQuantumKernel(
            feature_map=ZZFeatureMap(feature_dimension=X_sub.shape[1])
        )
        svc = SVC(kernel="linear", random_state=42)
        qsvc = SVC(kernel="precomputed", random_state=42)

        for train_idx, test_idx in rs.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            svc.fit(X_train, y_train)
            kernel_train = qkernel.evaluate_kernel(X_train)
            qsvc.fit(kernel_train, y_train)

            y_pred = svc.predict(X_test)
            classical_scores.append(metrics.accuracy_score(y_test, y_pred))

            kernel_test = qkernel.evaluate_kernel(X_test, X_train)
            y_pred = qsvc.predict(kernel_test)
            quantum_scores.append(metrics.accuracy_score(y_test, y_pred))

        avg_classical_score = sum(classical_scores) / len(classical_scores)
        avg_classical_scores.append(avg_classical_score)

        avg_quantum_score = sum(quantum_scores) / len(quantum_scores)
        avg_quantum_scores.append(avg_quantum_score)

    print(avg_classical_scores)
    print(avg_quantum_scores)
