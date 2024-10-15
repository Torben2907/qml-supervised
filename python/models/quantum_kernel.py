import numpy as np
from typing import Optional
import logging
from abc import abstractmethod, ABC

from pennylane import Operation, IQPEmbedding, QubitStateVector, device, qnode, adjoint
import pennylane as qml


class QuantumKernel(ABC):
    def __init__(self, *, feature_map: Operation) -> None:
        if feature_map is None:
            feature_map = IQPEmbedding(2)

        self.num_features = feature_map.num_params
        self._feature_map = feature_map

    @abstractmethod
    def evaluate_kernel(
        self, psi_vec: np.ndarray, phi_vec: Optional[np.ndarray] = None
    ):
        if phi_vec is None:
            logging.warning(
                "you passed only one data vector to the"
                + f"kernel computation, i. e. evaluating self inner product of {psi_vec}."
            )
        raise NotImplementedError()

    @property
    def num_features(self) -> int:
        return self.num_features

    @property
    def feature_map(self) -> Operation:
        return self._feature_map

    def _validate_inputs(self, psi_vec: np.ndarray, phi_vec: np.ndarray):
        pass


class FidelityStateVectorKernel(QuantumKernel):
    def __init__(
        self,
        *,
        feature_map: Operation,
        state_vector: QubitStateVector,
        shots: int,
        quantum_device: str = "default.qubit",
        num_qubits: int,
        num_repeats: int,
    ):
        super().__init__(feature_map=feature_map)
        self.state_vector = state_vector
        self.shots = shots
        self.quantum_device = quantum_device
        self.num_qubits = num_qubits
        self.num_repeats = num_repeats

    def evaluate_kernel(self, psi_vec, phi_vec=None):
        # concatenate vectors
        if phi_vec is not None:
            Z = np.array(
                [
                    np.concatenate((psi_vec[i], phi_vec[j]))
                    for i in range(len(psi_vec))
                    for j in range(len(phi_vec))
                ]
            )
        else:
            raise ValueError("cannot compute fidelity if only one argument is given.")

        self.build_circuit()
        kernel_vals = self.circuit(Z)[:, 0]
        kernel_matrix = np.reshape(kernel_vals, (len(psi_vec), len(phi_vec)))
        return kernel_matrix

    def build_circuit(self):
        dev = qml.device(self.quantum_device, wires=self.num_qubits)

        @qml.qnode(dev)
        def circuit(x):
            self.feature_map(
                x[: self.num_qubits],
                wires=range(self.num_qubits),
                n_repeats=self.num_repeats,
            )
            qml.adjoint(
                self.feature_map(
                    x[: self.num_qubits],
                    wires=range(self.num_qubits),
                    n_repeats=self.num_repeats,
                )
            )
            return qml.probs()

        self.circuit = circuit

        return circuit

    def _compute_fidelity(self, psi: np.ndarray, phi: np.ndarray):
        return np.abs(np.dot(np.conjugate(psi), phi)) ** 2
