import numpy as np
import copy
from typing import Optional
from sklearn.svm import SVC
from .quantum_kernel import QuantumKernel
from .qfm import QuantumFeatureMap
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_algorithms.utils import algorithm_globals


class QSVC(SVC):
    def __init__(
        self,
        *,
        quantum_kernel: Optional[QuantumKernel] = None,
        num_qubits: int = 2,
        num_layers: int = 2,
        feature_map: list[str] = ["RZ", "CZ"],
        entanglement: str = "linear",
        quantum_backend=None,
        random_state=None,
        **kwargs,
    ):
        SVC.__init__(self, **kwargs)

        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.feature_map = feature_map
        self.entanglement = entanglement
        self.random_state = random_state

        # simulate classically
        if not quantum_backend:
            np.random.seed(self.random_state)
            algorithm_globals._random_seed = self.random_state
            self.backend = AerSimulator(
                method="statevector",
                backend_options={
                    "method": "automatic",
                    "max_parallel_threads": 0,
                    "max_parallel_experiments": 0,
                    "max_parallel_shots": 0,
                },
            )

    def fit(self, X, y):
        num_samples, num_features = X.shape
        if isinstance(self.feature_map, list):
            self._fm = QuantumFeatureMap(
                num_features=num_features,
                num_qubits=self.num_qubits,
                num_layers=self.num_layers,
                gates=[g.upper() for g in self.feature_map],
                entanglement=self.entanglement,
                repeat=True,
                scale=False,
            )
        elif isinstance(self.feature_map, QuantumCircuit):
            self._fm = copy.deepcopy(self.feature_map)

        self.kernel = QuantumKernel(
            self._fm, quantum_instance=self.backend
        ).evaluate_kernel

        SVC.fit(self, X, y)

        return self

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    @property
    def quantum_kernel(self) -> QuantumKernel:
        return self._quantum_kernel
