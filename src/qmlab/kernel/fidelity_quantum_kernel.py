from typing import List, Tuple
import numpy as np
from .quantum_kernel import QuantumKernel
from qiskit import QuantumCircuit
from qiskit_algorithms.state_fidelities import BaseStateFidelity, ComputeUncompute
from qiskit.primitives import Sampler


class FidelityQuantumKernel(QuantumKernel):
    def __init__(
        self,
        *,
        feature_map: QuantumCircuit | None = None,
        fidelity: BaseStateFidelity | None = None,
        evaluate_duplicates: str = "off_diagonal",
        max_circuits_per_job: int | None = None,
        enforce_psd: bool = True,
    ):
        super().__init__(feature_map=feature_map, enforce_psd=enforce_psd)
        if not fidelity:
            fidelity = ComputeUncompute(sampler=Sampler())
        self._fidelity = fidelity
        self.max_circuits_per_job = max_circuits_per_job
        evaluate_duplicates = evaluate_duplicates.lower()
        if evaluate_duplicates not in ("all", "off_diagonal", "none"):
            raise ValueError(
                f"Value {evaluate_duplicates} isn't supported for attribute `eval_duplicates`!"
            )
        self._evaluate_duplicates = evaluate_duplicates

    def evaluate_kernel(self, psi_vec: np.ndarray, phi_vec: np.ndarray | None = None):
        psi_vec, phi_vec = self._validate_inputs(psi_vec, phi_vec)
        is_symmetric = phi_vec is None or np.array_equal(psi_vec, phi_vec)
        gram_matrix_shape = (
            len(psi_vec),
            len(phi_vec) if phi_vec is not None else len(psi_vec),
        )

        if is_symmetric:
            params = self._get_symmetric_parameterization(psi_vec)
            gram_matrix = self._compute_symmetric_kernel(gram_matrix_shape, *params)
            if self._enforce_psd:
                gram_matrix = self._ensure_psd(gram_matrix)
        else:
            if phi_vec is not None:
                params = self._get_parameterization(psi_vec, phi_vec)
                gram_matrix = self._compute_kernel(gram_matrix_shape, *params)
            else:
                raise ValueError("phi_vec cannot be None in the non-symmetric case!")

        return gram_matrix

    def _get_parameterization(
        self, psi_vec: np.ndarray, phi_vec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Combines x_vec and y_vec to get all the combinations needed to evaluate the kernel entries.
        """
        indices = [
            (i, j)
            for i, x_i in enumerate(psi_vec)
            for j, y_j in enumerate(phi_vec)
            if not self._is_trivial(i, j, x_i, y_j, False)
        ]

        return (
            psi_vec[[i for i, _ in indices]],
            phi_vec[[j for _, j in indices]],
            indices,
        )

    def _get_symmetric_parameterization(
        self, psi_vec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:

        indices = [
            (i, i + j)
            for i, x_i in enumerate(psi_vec)
            for j, x_j in enumerate(psi_vec[i:])
            if not self._is_trivial(i, i + j, x_i, x_j, True)
        ]

        return (
            psi_vec[[i for i, _ in indices]],
            psi_vec[[j for _, j in indices]],
            indices,
        )

    def _compute_kernel(
        self,
        kernel_shape: tuple[int, int],
        left_parameters: np.ndarray,
        right_parameters: np.ndarray,
        indices: List[Tuple[int, int]],
    ) -> np.ndarray:
        entries = self._compute_gram_entries(left_parameters, right_parameters)
        kernel_matrix = np.ones(kernel_shape)
        for i, (col, row) in enumerate(indices):
            kernel_matrix[col, row] = entries[i]
        return kernel_matrix

    def _compute_symmetric_kernel(
        self,
        gram_matrix_shape: tuple[int, int],
        left_parameters: np.ndarray,
        right_parameters: np.ndarray,
        indices: List[Tuple[int, int]],
    ) -> np.ndarray:
        """
        Given a set of parameterization, this computes the kernel matrix.
        """
        entries = self._compute_gram_entries(left_parameters, right_parameters)
        kernel_matrix = np.ones(gram_matrix_shape)
        for i, (col, row) in enumerate(indices):
            kernel_matrix[col, row] = entries[i]
            kernel_matrix[row, col] = entries[i]
        return kernel_matrix

    def _compute_gram_entries(
        self, left_parameters: np.ndarray, right_parameters: np.ndarray
    ) -> List[float]:
        if not left_parameters.size:
            return []

        entries = []
        if self.max_circuits_per_job:
            for start in range(0, len(left_parameters), self.max_circuits_per_job):
                end = start + self.max_circuits_per_job
                entries.extend(
                    self._run_fidelity(
                        left_parameters[start:end], right_parameters[start:end]
                    )
                )
        else:
            entries = self._run_fidelity(left_parameters, right_parameters)
        return entries

    def _run_fidelity(
        self, left_parameters: np.ndarray, right_parameters: np.ndarray
    ) -> List[float]:
        job = self.fidelity.run(
            [self.feature_map] * len(left_parameters),
            [self.feature_map] * len(left_parameters),
            left_parameters,
            right_parameters,
        )
        return job.result().fidelities

    def _is_trivial(
        self, i: int, j: int, psi_i: np.ndarray, phi_j: np.ndarray, symmetric: bool
    ) -> bool:
        if self._evaluate_duplicates == "all":
            return False
        if symmetric and i == j and self._evaluate_duplicates == "off_diagonal":
            return True
        if np.array_equal(psi_i, phi_j) and self._evaluate_duplicates == "none":
            return True
        return False

    @property
    def fidelity(self):
        """Returns the fidelity primitive used by this kernel."""
        return self._fidelity

    @property
    def evaluate_duplicates(self):
        """Returns the strategy used by this kernel to evaluate kernel matrix elements if duplicate
        samples are found."""
        return self._evaluate_duplicates
