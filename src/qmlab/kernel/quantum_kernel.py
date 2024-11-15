import numpy as np
import logging
from abc import abstractmethod, ABC
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap


class QuantumKernel(ABC):
    def __init__(
        self, *, feature_map: QuantumCircuit = None, enforce_psd: bool = True
    ) -> None:
        if feature_map is None:
            feature_map = ZZFeatureMap(2)

        self._feature_dimension = feature_map.feature_dimension
        self._feature_map = feature_map
        self._enforce_psd = enforce_psd

    @abstractmethod
    def evaluate_kernel(
        self, psi_vec: np.ndarray, phi_vec: np.ndarray | None = None
    ) -> np.ndarray:
        if phi_vec is None:
            logging.warning(
                "You've passed one state vector to the"
                + f"kernel computation, i. e. evaluating self inner product of {psi_vec} with itself."
            )
        raise NotImplementedError(
            "You're trying to call an abstract method of the base quantum kernel class."
        )

    @property
    def feature_map(self) -> QuantumCircuit:
        return self._feature_map

    @property
    def feature_dimension(self) -> int:
        return self._feature_dimension

    def _validate_inputs(self, psi_vec: np.ndarray, phi_vec: np.ndarray | None = None):
        if not isinstance(psi_vec, np.ndarray):
            raise ValueError(
                f"Data must be given as np.ndarray but has type {type(psi_vec)}!"
            )

        if psi_vec.ndim > 2:
            raise ValueError(
                f"{psi_vec} must be a one or two-dimensional array but has size {psi_vec.ndim}!"
            )

        if psi_vec.ndim == 1:
            psi_vec = psi_vec.reshape(-1, len(psi_vec))

        if psi_vec.shape[1] != self._feature_dimension:
            try:
                self._feature_map.num_qubits = psi_vec.shape[1]
            except AttributeError as ae:
                raise ValueError(
                    f"Incompatible dimensions found between {psi_vec} and class {self._feature_map.name}."
                    f"{psi_vec} has {psi_vec.shape[1]} but {self._feature_map.name} has "
                    f"{self._feature_map.num_qubits}."
                ) from ae

        if phi_vec is not None:
            if not isinstance(phi_vec, np.ndarray):
                raise ValueError(
                    f"Data must be given as np.ndarray but has type {type(phi_vec)}!"
                )

            if phi_vec.ndim > 2:
                raise ValueError(
                    f"{phi_vec} must be a one or two-dimensional array but has size {phi_vec.ndim}!"
                )

            if phi_vec.ndim == 1:
                phi_vec = phi_vec.reshape(-1, len(phi_vec))

        return psi_vec, phi_vec

    def _ensure_psd(self, kernel_matrix: np.ndarray) -> np.ndarray:
        r"""
        Find the closest positive semi-definite approximation to a symmetric kernel matrix.
        The (symmetric) matrix should always be positive semi-definite by construction,
        but this can be violated in case of noise, such as sampling noise.

        Args:
            kernel_matrix: Symmetric 2D array of the kernel entries.

        Returns:
            The closest positive semi-definite matrix.
        """
        w, v = np.linalg.eig(kernel_matrix)
        m = v @ np.diag(np.maximum(0, w)) @ v.transpose()
        return m.real
