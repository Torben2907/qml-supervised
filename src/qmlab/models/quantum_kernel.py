import numpy as np
from typing import Optional
import logging
from abc import abstractmethod, ABC

import pennylane as qml

from pennylane import IQPEmbedding
from pennylane.operation import Operation


class QuantumKernel(ABC):
    def __init__(self, *, feature_map: Operation) -> None:
        if feature_map is None:
            feature_map = IQPEmbedding(2)

        self._num_features = feature_map.num_params
        self._feature_map = feature_map

    @abstractmethod
    def build_circuit(self):
        raise NotImplementedError()

    @abstractmethod
    def evaluate_kernel(
        self, psi_vec: np.ndarray, phi_vec: Optional[np.ndarray] = None
    ):
        if phi_vec is None:
            logging.warning(
                "you passed one state vector to the"
                + f"kernel computation, i. e. evaluating self inner product of {psi_vec}."
            )
        # evaluation will be implemented for each kernel individually
        raise NotImplementedError()

    @property
    def num_features(self) -> int:
        return self.num_features

    @property
    def feature_map(self) -> Operation:
        return self._feature_map

    def _validate_inputs(self, psi_vec: np.ndarray, phi_vec: np.ndarray):
        if psi_vec.ndim > 2:
            raise ValueError(f"{psi_vec} must be a 2-dim array")

        if psi_vec.ndim == 1:
            psi_vec = psi_vec.reshape(-1, len(psi_vec))

        if phi_vec:
            if phi_vec.ndim > 2:
                raise ValueError(f"{phi_vec} must a a 2-dim array")

            if phi_vec.ndim == 1:
                phi_vec = phi_vec.reshape(-1, len(phi_vec))


class FidelityQuantumKernel(QuantumKernel):
    def __init__(
        self,
        *,
        feature_map: Operation,
        quantum_device: str = "default.qubit",
        shots: Optional[int] = None,
        num_qubits: int,
        num_repeats: int,
    ):
        super().__init__(feature_map=feature_map)
        self.shots = shots
        self.quantum_device = quantum_device
        self.num_qubits = num_qubits
        self.num_repeats = num_repeats
        self._circuit = None

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
            raise ValueError(
                "cannot compute the fidelity if only one argument is given."
            )

        self.build_circuit()
        kernel_vals = self._circuit(Z)[:, 0]
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
                    x[self.num_qubits :],
                    wires=range(self.num_qubits),
                    n_repeats=self.num_repeats,
                )
            )
            return qml.probs()

        self._circuit = circuit
        return circuit
