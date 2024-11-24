import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pennylane as qml
from .qmlab_testcase import QMLabTest
from qmlab.kernel import QSVC
from qmlab.kernel import FidelityQuantumKernel
from qmlab.data_generation import generate_random_data


class TestQSVCIntegration(QMLabTest):

    def test_classification_with_iris_data(self) -> None:
        qkernel = FidelityQuantumKernel(data_embedding=qml.IQPEmbedding)
        qsvm = QSVC(quantum_kernel=qkernel, random_state=self.random_state)

        X, y = load_iris(return_X_y=True)

        # make it a binary classification problem,
        # i.e. remove`virginica`-class.
        X = X[:100]
        y = y[:100]

        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        y_scaled = 2 * y - 1

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled)

        qsvm.fit(X_train, y_train)

        score = qsvm.score(X_test, y_test)

        assert score == 1.0

    def test_classification_with_random_data(self) -> None:
        qkernel = FidelityQuantumKernel(data_embedding=qml.IQPEmbedding)
        qsvm = QSVC(quantum_kernel=qkernel, random_state=self.random_state)

        X_train, y_train, X_test, y_test = generate_random_data(
            feature_dimension=2,
            training_examples_per_class=20,
            delta=0.3,
            test_examples_per_class=5,
            random_state=12345,
        )

        qsvm.fit(np.array(X_train), np.array(y_train))

        score = qsvm.score(np.array(X_test), np.array(y_test))

        assert score > 0.5
