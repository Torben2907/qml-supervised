import jax
import sklearn
import pytest
import numpy as np
import jax.numpy as jnp
import pennylane as qml
from qmlab.kernel.angle_embedded_kernel import QSVC
from pennylane import QNode
from qmlab_testcase import QMLabTest


class TestQSVC(QMLabTest):

    def test_qsvc_is_abstract(self):
        with pytest.raises(TypeError):
            QSVC()

    def test_qsvc_initialization(self):
        qsvm = DummyQSVC()

        assert isinstance(qsvm.svm, sklearn.svm.SVC)
        assert qsvm.svm.kernel == "precomputed"
        assert qsvm.reps == 2
        assert qsvm.C == 1.0

    def test_qsvc_create_random_key(self):
        qsvm = DummyQSVC()
        random_key = qsvm.create_random_key()

        assert isinstance(random_key, jnp.ndarray)

    def test_qsvc_build_circuit(self):
        circuit = DummyQSVC().build_circuit()
        assert callable(circuit)

    def test_qsvc_fit_and_predict(self):
        qsvc = DummyQSVC()
        X = np.array([[0, 1], [1, 0]])
        y = np.array([0, 1])

        qsvc.fit(X, y)
        predictions = qsvc.predict(X)

        assert len(predictions) == len(X)
        assert (predictions == 0).all()  # Predict always 0 in this mock


class DummyQSVC(QSVC):

    def build_circuit(self) -> QNode:
        dev = qml.device(self.dev_type, wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

        if self.jit is True:
            circuit = jax.jit(circuit)

        return circuit

    def initialize_params(
        self, feature_dimension: int | None, class_labels: np.ndarray | None
    ) -> None:
        self.build_circuit()

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
        return np.ones_like(x_vec)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QSVC":
        self.initialize_params(X.shape[1], np.unique(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full((len(X), 2), 0.5)
