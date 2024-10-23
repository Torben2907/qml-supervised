from typing import Optional
from sklearn.svm import SVC
from .quantum_kernel import QuantumKernel, FidelityStateVectorKernel


class QSVC(SVC):
    def __init__(self, *, quantum_kernel: Optional[QuantumKernel] = None, **kwargs):
        self.quantum_kernel = (
            quantum_kernel if quantum_kernel else FidelityStateVectorKernel
        )

    @property
    def quantum_kernel(self) -> QuantumKernel:
        return self.quantum_kernel

    @quantum_kernel.setter
    def set_quantum_kernel(self, quantum_kernel: QuantumKernel):
        self.quantum_kernel = quantum_kernel
        self.kernel = self.quantum_kernel.evaluate_kernel
