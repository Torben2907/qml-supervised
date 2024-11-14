import sys

sys.path.append("./src")

import pennylane as qml
import numpy as np
from qmlab.models.quantum_kernel import FidelityQuantumKernel
from qmlab.models.qsvc import QSVC
from qmlab.preprocessing import (
    parse_biomed_data_to_ndarray,
    scale_to_specified_range,
)

X, y = parse_biomed_data_to_ndarray("haberman_new")
_, num_features = X.shape
X = scale_to_specified_range(X, range=(-np.pi / 2, np.pi / 2))


quantum_kernel = FidelityQuantumKernel(
    feature_map=qml.IQPEmbedding, shots=None, num_qubits=num_features, num_repeats=0
)

qsvm = QSVC(quantum_kernel=quantum_kernel)
qsvm.fit(X, y)
