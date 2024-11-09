import numpy as np
from typing import Optional
from sklearn.svm import SVC
from .quantum_kernel import QuantumKernel, FidelityQuantumKernel
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_aer import AerSimulator


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

        self._quantum_kernel = (
            quantum_kernel if quantum_kernel else FidelityQuantumKernel
        )

        if not quantum_backend:
            np.random.seed(self.random_state)
            algorithm_globals.random_seed = self.random_state
            self.backend = Quant

        if not random_state:
            self.random_state = set_global_seed(42)

    @property
    def quantum_kernel(self) -> QuantumKernel:
        return self._quantum_kernel

    @quantum_kernel.setter
    def set_quantum_kernel(self, quantum_kernel: QuantumKernel):
        self._quantum_kernel = quantum_kernel
        self.kernel = self._quantum_kernel.evaluate_kernel
