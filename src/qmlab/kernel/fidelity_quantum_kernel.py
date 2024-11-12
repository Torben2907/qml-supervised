from typing import List, Sequence, Tuple
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
        eval_duplicates: str = "off_diagonal",
        max_circuits_per_job: int | None = None,
        enforce_psd: bool = True,
    ):
        super().__init__(feature_map=feature_map, enforce_psd=enforce_psd)
        if not fidelity:
            fidelity = ComputeUncompute(sampler=Sampler())
        self._fidelity = fidelity
        self.max_circuits_per_job = max_circuits_per_job
        eval_duplicates = eval_duplicates.lower()
        if eval_duplicates not in ("all", "off_diagonal", "none"):
            raise ValueError(
                f"Value {eval_duplicates} isn't supported for attribute `eval_duplicates`!"
            )
        self._evaluate_duplicates = eval_duplicates

    def evaluate_kernel(self, psi_vec: np.ndarray, phi_vec: np.ndarray | None = None):
        psi_vec, phi_vec = self._validate_inputs(psi_vec, phi_vec)

        is_symmetric = True
        if phi_vec is None:
            phi_vec = psi_vec
        elif not np.array_equal(psi_vec, phi_vec):
            is_symmetric = False

        gram_matrix_shape = (psi_vec.shape[0], phi_vec.shape[0])

        if is_symmetric:
            left_parameters, right_parameters, indices = (
                self._get_symmetric_parameterization(psi_vec)
            )
            gram_matrix = self._get_symmetric_kernel_matrix(
                gram_matrix_shape, left_parameters, right_parameters, indices
            )
        else:
            left_parameters, right_parameters, indices = self._get_parameterization(
                psi_vec, phi_vec
            )
            gram_matrix = self._get_kernel_matrix(
                gram_matrix_shape, left_parameters, right_parameters, indices
            )

        if is_symmetric and self._enforce_psd:
            gram_matrix = self._ensure_psd(gram_matrix)

        return gram_matrix

    def _get_parameterization(
        self, psi_vec: np.ndarray, phi_vec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Combines x_vec and y_vec to get all the combinations needed to evaluate the kernel entries.
        """
        num_features = psi_vec.shape[1]
        left_parameters = np.zeros((0, num_features))
        right_parameters = np.zeros((0, num_features))

        indices = np.asarray(
            [
                (i, j)
                for i, x_i in enumerate(psi_vec)
                for j, y_j in enumerate(phi_vec)
                if not self._is_trivial(i, j, x_i, y_j, False)
            ]
        )

        if indices.size > 0:
            left_parameters = psi_vec[indices[:, 0]]
            right_parameters = phi_vec[indices[:, 1]]

        return left_parameters, right_parameters, indices.tolist()

    def _get_symmetric_parameterization(
        self, psi_vec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Combines two copies of x_vec to get all the combinations needed to evaluate the kernel entries.
        """
        num_features = psi_vec.shape[1]
        left_parameters = np.zeros((0, num_features))
        right_parameters = np.zeros((0, num_features))

        indices = np.asarray(
            [
                (i, i + j)
                for i, x_i in enumerate(psi_vec)
                for j, x_j in enumerate(psi_vec[i:])
                if not self._is_trivial(i, i + j, x_i, x_j, True)
            ]
        )

        if indices.size > 0:
            left_parameters = psi_vec[indices[:, 0]]
            right_parameters = psi_vec[indices[:, 1]]

        return left_parameters, right_parameters, indices.tolist()

    def _get_kernel_matrix(
        self,
        kernel_shape: tuple[int, int],
        left_parameters: np.ndarray,
        right_parameters: np.ndarray,
        indices: List[Tuple[int, int]],
    ) -> np.ndarray:
        """
        Given a parameterization, this computes the symmetric kernel matrix.
        """
        kernel_entries = self._get_kernel_entries(left_parameters, right_parameters)

        # fill in trivial entries and then update with fidelity values
        kernel_matrix = np.ones(kernel_shape)

        for i, (col, row) in enumerate(indices):
            kernel_matrix[col, row] = kernel_entries[i]

        return kernel_matrix

    def _get_symmetric_kernel_matrix(
        self,
        gram_matrix_shape: tuple[int, int],
        left_parameters: np.ndarray,
        right_parameters: np.ndarray,
        indices: List[Tuple[int, int]],
    ) -> np.ndarray:
        """
        Given a set of parameterization, this computes the kernel matrix.
        """
        kernel_entries = self._get_kernel_entries(left_parameters, right_parameters)
        kernel_matrix = np.ones(gram_matrix_shape)

        for i, (col, row) in enumerate(indices):
            kernel_matrix[col, row] = kernel_entries[i]
            kernel_matrix[row, col] = kernel_entries[i]

        return kernel_matrix

    def _get_kernel_entries(
        self, left_parameters: np.ndarray, right_parameters: np.ndarray
    ) -> Sequence[float]:
        """
        Gets kernel entries by executing the underlying fidelity instance and getting the results
        back from the async job.
        """
        num_circuits = left_parameters.shape[0]
        kernel_entries = []
        # Check if it is trivial case, only identical samples
        if num_circuits != 0:
            if self.max_circuits_per_job is None:
                job = self._fidelity.run(
                    [self._feature_map] * num_circuits,
                    [self._feature_map] * num_circuits,
                    left_parameters,
                    right_parameters,
                )
                kernel_entries = job.result().fidelities
            else:
                # Determine the number of chunks needed
                num_chunks = (
                    num_circuits + self.max_circuits_per_job - 1
                ) // self.max_circuits_per_job
                for i in range(num_chunks):
                    # Determine the range of indices for this chunk
                    start_idx = i * self.max_circuits_per_job
                    end_idx = min((i + 1) * self.max_circuits_per_job, num_circuits)
                    # Extract the parameters for this chunk
                    chunk_left_parameters = left_parameters[start_idx:end_idx]
                    chunk_right_parameters = right_parameters[start_idx:end_idx]
                    # Execute this chunk
                    job = self._fidelity.run(
                        [self._feature_map] * (end_idx - start_idx),
                        [self._feature_map] * (end_idx - start_idx),
                        chunk_left_parameters,
                        chunk_right_parameters,
                    )
                    # Extend the kernel_entries list with the results from this chunk
                    kernel_entries.extend(job.result().fidelities)
        return kernel_entries

    def _is_trivial(
        self, i: int, j: int, psi_i: np.ndarray, phi_j: np.ndarray, symmetric: bool
    ) -> bool:
        """
        Verifies if the kernel entry is trivial (to be set to `1.0`) or not.

        Args:
            i: row index of the entry in the kernel matrix.
            j: column index of the entry in the kernel matrix.
            x_i: a sample from the dataset that corresponds to the row in the kernel matrix.
            y_j: a sample from the dataset that corresponds to the column in the kernel matrix.
            symmetric: whether it is a symmetric case or not.

        Returns:
            `True` if the entry is trivial, `False` otherwise.
        """
        # if we evaluate all combinations, then it is non-trivial
        if self._evaluate_duplicates == "all":
            return False

        # if we are on the diagonal and we don't evaluate it, it is trivial
        if symmetric and i == j and self._evaluate_duplicates == "off_diagonal":
            return True

        # if don't evaluate any duplicates
        if np.array_equal(psi_i, phi_j) and self._evaluate_duplicates == "none":
            return True

        # otherwise evaluate
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
