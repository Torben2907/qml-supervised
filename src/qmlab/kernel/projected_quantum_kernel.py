import jax
import jax.numpy as jnp
import pennylane as qml
import numpy as np
from typing import List
from .quantum_kernel import QuantumKernel
from pennylane.operation import Operation
from .kernel_utils import vmap_batch


class ProjectedQuantumKernel(QuantumKernel):

    def __init__(
        self,
        *,
        data_embedding: Operation | str = None,
        device_type: str = "default.qubit",
        reps: int = 2,
        rotation: str | None = "Z",
        enforce_psd: bool = False,
        jit: bool = True,
        max_batch_size: int = 256,
        interface: str = "jax",
        trotter_steps: int = 10,
    ) -> None:
        self._data_embedding = self.initialize_embedding(data_embedding)
        self.trotter_steps = trotter_steps
        self._jit = jit

    def initialize_params(
        self, feature_dimension: int, class_labels: List[int] | None = None
    ) -> None:
        if class_labels is None:
            class_labels = [-1, 1]

        self.classes_ = class_labels
        self.n_classes_ = len(self.classes_)
        assert self.n_classes_ == 2
        assert 1 in self.classes_ and -1 in self.classes_

        self.n_features_ = feature_dimension

        if self._data_embedding == "IQP":
            self.n_qubits_ = self.n_features_
        elif self._data_embedding == "Hamiltonian":
            self.n_qubits_ = self.n_features_ + 1
            self.rotation_angles_ = jnp.array(
                np.random.default_rng().uniform(size=(self.n_qubits_, 3)) * np.pi * 2
            )
        self.build_circuit()

    def build_circuit(self):
        """
        Constructs the circuit to get the expvals of a given qubit and Pauli operator
        We will use JAX to parallelize over these circuits in precompute kernel.
        Args:
            P: a pennylane Pauli X,Y,Z operator on a given qubit
        """
        if self.embedding == "IQP":

            def embedding(x):
                qml.IQPEmbedding(x, wires=range(self.n_qubits_), n_repeats=2)

        elif self.embedding == "Hamiltonian":

            def embedding(x):
                evol_time = self.t / self.trotter_steps * (self.n_qubits_ - 1)
                for i in range(self.n_qubits_):
                    qml.Rot(
                        self.rotation_angles_[i, 0],
                        self.rotation_angles_[i, 1],
                        self.rotation_angles_[i, 2],
                        wires=i,
                    )
                for __ in range(self.trotter_steps):
                    for j in range(self.n_qubits_ - 1):
                        qml.IsingXX(x[j] * evol_time, wires=[j, j + 1])
                        qml.IsingYY(x[j] * evol_time, wires=[j, j + 1])
                        qml.IsingZZ(x[j] * evol_time, wires=[j, j + 1])

        dev = qml.device(self.dev_type, wires=self.n_qubits_)

        @qml.qnode(dev, **self.qnode_kwargs)
        def circuit(x):
            embedding(x)
            return (
                [qml.expval(qml.PauliX(wires=i)) for i in range(self.n_qubits_)]
                + [qml.expval(qml.PauliY(wires=i)) for i in range(self.n_qubits_)]
                + [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits_)]
            )

        self.circuit = circuit

        def circuit_as_array(x):
            return jnp.array(circuit(x))

        if self._jit:
            circuit_as_array = jax.jit(circuit_as_array)
        circuit_as_array = jax.vmap(circuit_as_array, in_axes=(0))
        circuit_as_array = vmap_batch(circuit_as_array, 0, self.max_vmap)

        return circuit_as_array

    def evaluate(self, x, y):
        dim1 = len(x)
        dim2 = len(y)
        self.circuit = self.build_circuit()

        valsX1 = np.array(self.circuit(x))
        valsX1 = np.reshape(valsX1, (dim1, 3, -1))
        valsX2 = np.array(self.circuit(y))
        valsX2 = np.reshape(valsX2, (dim2, 3, -1))

        valsX_X1 = valsX1[:, 0]
        valsX_X2 = valsX2[:, 0]
        valsY_X1 = valsX1[:, 1]
        valsY_X2 = valsX2[:, 1]
        valsZ_X1 = valsX1[:, 2]
        valsZ_X2 = valsX2[:, 2]

        all_vals_X1 = np.reshape(np.concatenate((valsX_X1, valsY_X1, valsZ_X1)), -1)
        default_gamma = 1 / np.var(all_vals_X1) / self.n_features_

        # construct kernel following plots
        gram_matrix = np.zeros([dim1, dim2])

        for i in range(dim1):
            for j in range(dim2):
                sumX = sum(
                    [
                        (valsX_X1[i, q] - valsX_X2[j, q]) ** 2
                        for q in range(self.n_qubits_)
                    ]
                )
                sumY = sum(
                    [
                        (valsY_X1[i, q] - valsY_X2[j, q]) ** 2
                        for q in range(self.n_qubits_)
                    ]
                )
                sumZ = sum(
                    [
                        (valsZ_X1[i, q] - valsZ_X2[j, q]) ** 2
                        for q in range(self.n_qubits_)
                    ]
                )

                gram_matrix[i, j] = np.exp(
                    -default_gamma * self.gamma_factor * (sumX + sumY + sumZ)
                )
        return gram_matrix
