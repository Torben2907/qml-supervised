from typing import Dict
import jax
import numpy as np
from .quantum_kernel import QuantumKernel
from pennylane import QNode
from pennylane.operation import Operation
from pennylane.measurements import ProbabilityMP
import pennylane as qml
import jax.numpy as jnp
from qmlab.utils import chunk_vmapped_fn


class FidelityQuantumKernel(QuantumKernel):
    def __init__(
        self,
        *,
        embedding: Operation,
        device: str = "default.qubit",
        enforce_psd: bool = False,
        jit: bool = True,
        max_vmap: int = 250,
        evaluate_duplicates: str = "off_diagonal",
        qnode_kwargs: Dict[str, str | None] = {
            "interface": "jax-jit",
            "diff_method": None,
        },
    ):
        super().__init__(
            embedding=embedding,
            device=device,
            enforce_psd=enforce_psd,
            jit=jit,
            max_vmap=max_vmap,
            qnode_kwargs=qnode_kwargs,
        )
        evaluate_duplicates = evaluate_duplicates.lower()
        if evaluate_duplicates not in ("all", "off_diagonal", "none"):
            raise ValueError(
                f"Value {evaluate_duplicates} isn't supported for attribute `eval_duplicates`!"
            )
        self._evaluate_duplicates = evaluate_duplicates

    def build_circuit(self) -> QNode:

        @qml.qnode(self._device, **self._qnode_kwargs)
        def circuit(z: np.ndarray) -> ProbabilityMP:
            self._embedding(features=z[: self.num_qubits], wires=range(self.num_qubits))
            qml.adjoint(
                self._embedding(
                    features=z[self.num_qubits :], wires=range(self.num_qubits)
                )
            )
            return qml.probs()

        self.circuit = circuit
        if self._jit:
            circuit = jax.jit(circuit)

        return circuit

    def evaluate(self, x: np.ndarray, y: np.ndarray | None = None):
        x, y = self._validate_inputs(x, y)
        # is_symmetric = y is None or np.array_equal(x, y)
        kernel_matrix_shape = (
            len(x),
            len(y) if y is not None else len(x),
        )

        if y is not None:
            Z = jnp.array(
                [
                    np.concatenate((x[i], y[j]))
                    for i in range(len(x))
                    for j in range(len(y))
                ]
            )

        circuit = self.build_circuit()
        self.batched_circuit = chunk_vmapped_fn(
            jax.vmap(circuit, 0), start=0, max_vmap=self._max_vmap
        )

        # we are only interested in measuring |0>
        kernel_values = self.batched_circuit(Z)[:, 0]
        kernel_matrix = np.reshape(kernel_values, kernel_matrix_shape)

        if self._enforce_psd:
            kernel_matrix = self._make_psd(kernel_matrix)

        return kernel_matrix

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
    def evaluate_duplicates(self):
        return self._evaluate_duplicates
