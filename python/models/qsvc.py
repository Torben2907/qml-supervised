from typing import Optional
from sklearn.svm import SVC
from quantum_kernel import QuantumKernel


class QSVC(SVC):
    def __init__(self, *, quantum_kernel: Optional[QuantumKernel] = None, **kwargs):
        self.quantum_kernel = quantum_kernel
