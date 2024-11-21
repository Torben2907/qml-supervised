import jax
import numpy as np
import jax.numpy as jnp

import pennylane as qml
from pennylane.measurements import ProbabilityMP
from pennylane import QNode

from qmlab.utils import chunk_vmapped_fn
from .qsvm import QSVC


class AngleEmbeddedKernel(QSVC):

    def build_circuit(self) -> QNode:
        dev = qml.device(self.dev_type, wires=self.num_qubits)

        @qml.qnode(dev)
        def circuit(concat_vec: np.ndarray) -> ProbabilityMP:
            qml.AngleEmbedding(
                concat_vec[self.num_qubits :], wires=range(self.num_qubits)
            )
            qml.adjoint(qml.AngleEmbedding)(
                concat_vec[: self.num_qubits], wires=range(self.num_qubits)
            )
            return qml.probs()

        self.circuit = circuit
        if self.jit:
            circuit = jax.jit(circuit)
        return circuit

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
        Z = jnp.array(
            [
                np.concatenate((x_vec[i], y_vec[j]))
                for i in range(len(x_vec))
                for j in range(len(y_vec))
            ]
        )
        circuit = self.build_circuit()
        self.batched_circuit = chunk_vmapped_fn(
            jax.vmap(circuit, 0), start=0, max_vmap=self.max_vmap
        )
        kernel_vals = self.batched_circuit(Z)[:, 0]
        kernel_matrix = np.reshape(kernel_vals, (len(x_vec), len(y_vec)))
        return kernel_matrix

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AngleEmbeddedKernel":
        self.initialize_params(X.shape[1], np.unique(y))
        self.parameters = {"X_train": X}
        gram_matrix = self.evaluate(X, X)
        self.svm.fit(gram_matrix, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        test_train_kernel_matrix = self.evaluate(X, self.parameters["X_train"])
        return self.svm.predict(test_train_kernel_matrix)

    def predict_proba(self, X):
        self._check_fitted()
        kernel_matrix = self.evaluate(X, self.parameters["X_train"])
        return self.svm.predict_proba(kernel_matrix)
