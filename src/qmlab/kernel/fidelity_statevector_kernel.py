from __future__ import annotations

from functools import lru_cache
from typing import Type, TypeVar, Any

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_algorithms.utils import algorithm_globals


from .quantum_kernel import QuantumKernel

SV = TypeVar("SV", bound=Statevector)


class FidelityStatevectorKernel(QuantumKernel):

    def __init__(
        self,
        *,
        feature_map: QuantumCircuit | None = None,
        statevector_type: Type[SV] = Statevector,
        cache_size: int | None = None,
        auto_clear_cache: bool = True,
        shots: int | None = None,
        enforce_psd: bool = True,
    ) -> None:

        super().__init__(feature_map=feature_map)

        self._statevector_type = statevector_type
        self._auto_clear_cache = auto_clear_cache
        self._shots = shots
        self._enforce_psd = enforce_psd
        self._cache_size = cache_size
        # Create the statevector cache at the instance level.
        self._get_statevector = lru_cache(maxsize=cache_size)(self._get_statevector_)

    def evaluate_kernel(
        self,
        x_vec: np.ndarray,
        y_vec: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._auto_clear_cache:
            self.clear_cache()

        x_vec, y_vec = self._validate_inputs(x_vec, y_vec)

        # Determine if calculating self inner product.
        is_symmetric = True
        if y_vec is None:
            y_vec = x_vec
        elif not np.array_equal(x_vec, y_vec):
            is_symmetric = False

        return self._evaluate_kernel(x_vec, y_vec, is_symmetric)

    def _evaluate_kernel(
        self, x_vec: np.ndarray, y_vec: np.ndarray, is_symmetric: bool
    ):
        kernel_shape = (x_vec.shape[0], y_vec.shape[0])

        x_svs = [self._get_statevector(tuple(x)) for x in x_vec]
        y_svs = [self._get_statevector(tuple(y)) for y in y_vec]

        kernel_matrix = np.ones(kernel_shape)
        for i, x in enumerate(x_svs):
            for j, y in enumerate(y_svs):
                if np.array_equal(x, y):
                    continue
                kernel_matrix[i, j] = self._compute_kernel_entry(x, y)

        return kernel_matrix

    def _get_statevector_(self, param_values: tuple[float]) -> np.ndarray:
        # lru_cache requires hashable function arguments.
        qc = self._feature_map.assign_parameters(param_values)
        return self._statevector_type(qc).data

    def _compute_kernel_entry(self, x: np.ndarray, y: np.ndarray) -> float:
        fidelity = self._compute_fidelity(x, y)
        if self._shots is not None:
            fidelity = self._add_shot_noise(fidelity)
        return fidelity

    @staticmethod
    def _compute_fidelity(x: np.ndarray, y: np.ndarray) -> float:
        return np.abs(np.conj(x) @ y) ** 2

    def _add_shot_noise(self, fidelity: float) -> float:
        return (
            algorithm_globals.random.binomial(n=self._shots, p=fidelity) / self._shots
        )

    def clear_cache(self):
        """Clear the statevector cache."""
        self._get_statevector.cache_clear()

    def __getstate__(self) -> dict[str, Any]:
        kernel = dict(self.__dict__)
        kernel["_get_statevector"] = None
        return kernel

    def __setstate__(self, kernel: dict[str, Any]):
        self.__dict__ = kernel
        self._get_statevector = lru_cache(maxsize=self._cache_size)(
            self._get_statevector_
        )
