import numpy as np
import pennylane as qml
from qmlab.preprocessing import parse_biomed_data_to_ndarray
from qmlab.kernel import QSVC, FidelityQuantumKernel
from qmlab.utils import run_cross_validation

X, y, feature_names = parse_biomed_data_to_ndarray("haberman_new", return_X_y=True)
qkernel = FidelityQuantumKernel(embedding=qml.IQPEmbedding, jit=True)
qsvc = QSVC(quantum_kernel=qkernel)
print(run_cross_validation(qsvc, X, y))
