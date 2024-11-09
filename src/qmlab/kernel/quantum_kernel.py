import numpy as np
from typing import Optional
import logging
from abc import abstractmethod, ABC
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler


class QuantumKernel(ABC):
    def __init__(self, *, feature_map: QuantumCircuit = None) -> None:
        if feature_map is None:
            feature_map = ZZFeatureMap(2)

        self._num_features = feature_map.num_params
        self._feature_map = feature_map

    @abstractmethod
    def build_circuit(self):
        raise NotImplementedError()

    @abstractmethod
    def evaluate_kernel(
        self, psi_vec: np.ndarray, phi_vec: Optional[np.ndarray] = None
    ) -> np.ndarray:
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
    def feature_map(self) -> QuantumCircuit:
        return self._feature_map

    def _validate_inputs(
        self, psi_vec: np.ndarray, phi_vec: Optional[np.ndarray] = None
    ):
        if psi_vec.ndim > 2:
            raise ValueError(
                f"{psi_vec} must be a one or two-dimensional array but has size {psi_vec.ndim}!"
            )

        if psi_vec.ndim == 1:
            psi_vec = psi_vec.reshape(-1, len(psi_vec))

        if psi_vec.shape[1] != self._num_features:
            try:
                self._feature_map.num_qubits = psi_vec.shape[1]
            except AttributeError as ae:
                raise ValueError(
                    f"Incompatible dimensions found between {psi_vec} and class {self._feature_map.name}."
                    f"{psi_vec} has {psi_vec.shape[1]} but {self._feature_map.name} has "
                    f"{self._feature_map.num_qubits}."
                ) from ae

        if phi_vec:
            if phi_vec.ndim > 2:
                raise ValueError(
                    f"{phi_vec} must be a one or two-dimensional array but has size {phi_vec.ndim}!"
                )

            if phi_vec.ndim == 1:
                phi_vec = phi_vec.reshape(-1, len(phi_vec))
