import jax
import numpy as np

import pennylane as qml
from pennylane.measurements import ProbabilityMP
from pennylane import QNode

from qmlab.utils import chunk_vmapped_fn
from .qsvm import QSVC


class AngleEmbeddedKernel(QSVC):

    def build_circuit(self) -> QNode:
        dev = qml.device(self.dev_type, wires=self.num_qubits)

        projector = np.zeros((2**self.num_qubits, 2**self.num_qubits))
        projector[0, 0] = 1.0

        @qml.qnode(dev)
        def circuit(x1: np.ndarray, x2: np.ndarray) -> ProbabilityMP:
            qml.AngleEmbedding(x1, wires=range(self.num_qubits))
            qml.adjoint(qml.AngleEmbedding)(x2, wires=range(self.num_qubits))
            return qml.expval(qml.Hermitian(projector, wires=range(self.num_qubits)))

        self.circuit = circuit

        if self.jit:
            circuit = jax.jit(circuit)

        return circuit

    def initialize_params(
        self, feature_dimension: int, class_labels: np.ndarray
    ) -> None:
        if class_labels is None:
            class_labels = np.asarray([-1, 1])

        self.classes_ = class_labels
        self.num_classes = len(self.classes_)
        assert self.num_classes == 2
        assert -1 in self.classes_ and +1 in self.classes_
        self.num_qubits = feature_dimension

        self.build_circuit()

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
        circuit = self.build_circuit()
        self.batched_circuit = chunk_vmapped_fn(
            jax.vmap(circuit, 0), start=0, max_vmap=self.max_vmap
        )
        kernel_matrix = np.array(
            [[self.batched_circuit(a, b) for b in x_vec] for a in y_vec]
        )
        return kernel_matrix

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AngleEmbeddedKernel":
        self.initialize_params(X.shape[1], np.unique(y))
        self.parameters = {"X_train": X}
        gram_matrix = self.evaluate(X, X)
        self.svm.fit(gram_matrix, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        test_train_kernel_matrix = self.evaluate(X, self.parameters["X_train"])
        return self.svm.predict(test_train_kernel_matrix)
