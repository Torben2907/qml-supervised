import jax.numpy as jnp
import numpy as np
from typing import List
from .quantum_kernel import QuantumKernel
from pennylane.operation import Operation


class ProjectedQuantumKernel(QuantumKernel):

    def __init__(
        self,
        embedding: Operation | str = "Hamiltonian",
        jit: bool = True,
        max_vmap: int | None = 250,
        device: str = "default.qubit",
        interface: str = "jax_jit",
        gamma: float = 1.0,
    ) -> None:
        self._embedding = self.initialize_embedding(embedding)

    def initialize_params(
        self, feature_dimension: int, class_labels: List[int] | None = None
    ) -> None:
        if class_labels is None:
            class_labels = [-1, 1]

        self.classes_ = class_labels
        self.n_classes_ = len(self.classes_)
        assert self.n_classes_ == 2
        self.n_features_ = feature_dimension

        if self.data_embedding == "IQP":
            self.n_qubits_ = self.n_features_
        elif self.data_embedding == "Hamiltonian":
            self.n_qubits_ = self.n_features_ + 1
            self.rotation_angles_ = jnp.array(
                np.random.default_rng().uniform(size=(self.n_qubits_, 3)) * np.pi * 2
            )

    def build_circuit(self):
        pass

    def evaluate(self, x, y):
        pass
