from typing import List
import jax
import numpy as np
import jax.numpy as jnp
import pennylane as qml
from .quantum_kernel import QuantumKernel
from pennylane import QNode
from pennylane.operation import Operation
from pennylane.measurements import ProbabilityMP
from .kernel_utils import vmap_batch
from ..exceptions import InvalidEmbeddingError, QMLabError


class FidelityQuantumKernel(QuantumKernel):
    def __init__(
        self,
        *,
        data_embedding: Operation | str,
        device: str = "default.qubit",
        enforce_psd: bool = False,
        jit: bool = True,
        max_vmap: int = 250,
        evaluate_duplicates: str = "off_diagonal",
        interface: str = "jax-jit",
    ):
        super().__init__(
            data_embedding=data_embedding,
            device_type=device,
            enforce_psd=enforce_psd,
            jit=jit,
            max_vmap=max_vmap,
            interface=interface,
        )
        evaluate_duplicates = evaluate_duplicates.lower()
        if evaluate_duplicates not in ("all", "off_diagonal", "none"):
            raise ValueError(
                f"Value {evaluate_duplicates} isn't supported for attribute `eval_duplicates`!"
            )
        self._evaluate_duplicates = evaluate_duplicates

    def initialize(
        self,
        feature_dimension: int,
        class_labels: List[int] | None = None,
    ) -> None:
        if class_labels is None:
            class_labels = [-1, 1]

        self.classes_ = class_labels
        self.n_classes_ = len(self.classes_)
        assert +1 and -1 in self.classes_
        assert self.n_classes_ == 2

        if (
            self._data_embedding == qml.IQPEmbedding
            or self._data_embedding == qml.AngleEmbedding
        ):
            self.num_qubits = feature_dimension
        elif self._data_embedding == qml.AmplitudeEmbedding:
            self.num_qubits = int(np.ceil(np.log2(feature_dimension)))
        else:
            raise InvalidEmbeddingError("Invalid embedding. Stop.")

    def build_circuit(self) -> QNode:
        self.device = qml.device(self._device_type, wires=self.num_qubits)

        @qml.qnode(self.device, interface=self.interface, diff_method=None)
        def circuit(concat_vec: jnp.ndarray) -> ProbabilityMP:
            if self.num_qubits is None:
                raise QMLabError(
                    "Number of qubits has not been specified before building the circuit!"
                )
            # noinspection PyCallingNonCallable
            self._data_embedding(
                features=concat_vec[: self.num_qubits], wires=range(self.num_qubits)
            )
            # noinspection PyCallingNonCallable
            qml.adjoint(
                self._data_embedding(
                    features=concat_vec[self.num_qubits :], wires=range(self.num_qubits)
                )
            )
            return qml.probs()

        self.circuit = circuit
        if self._jit:
            circuit = jax.jit(circuit)

        return circuit

    def evaluate(self, x: np.ndarray, y: np.ndarray):
        x, y = self._validate_inputs(x, y)
        # is_symmetric = y is None or np.array_equal(x, y)
        kernel_matrix_shape = (
            len(x),
            len(y) if y is not None else len(x),
        )

        Z = jnp.array(
            [np.concatenate((x[i], y[j])) for i in range(len(x)) for j in range(len(y))]
        )

        circuit = self.build_circuit()
        self.batched_circuit = vmap_batch(
            jax.vmap(circuit, 0), start=0, max_vmap=self._max_vmap
        )

        # we are only interested in measuring |0>
        kernel_values = self.batched_circuit(Z)[:, 0]
        kernel_matrix = np.reshape(kernel_values, kernel_matrix_shape)

        if self._enforce_psd:
            kernel_matrix = self.make_psd(kernel_matrix)

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
    def evaluate_duplicates(self) -> str:
        return self._evaluate_duplicates

    @evaluate_duplicates.setter
    def evaluate_duplicates(self, evaluate_duplicates: str) -> None:
        self._evaluate_duplicates = evaluate_duplicates
