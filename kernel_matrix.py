import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qmlab.kernel.fidelity_quantum_kernel import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute

# Instantiate a feature map
feature_map = ZZFeatureMap(feature_dimension=2, reps=2)

# Create a FidelityQuantumKernel instance
quantum_kernel = FidelityQuantumKernel(
    feature_map=feature_map,
    fidelity=ComputeUncompute(sampler=Sampler()),
    evaluate_duplicates="off_diagonal",
    max_circuits_per_job=100,
)

psi_vec = np.array([[0.1, 0.2], [0.3, 0.4]])
phi_vec = np.array([[0.5, 0.6], [0.7, 0.8]])

# Compute kernel matrix
kernel_matrix = quantum_kernel.evaluate_kernel(psi_vec, psi_vec)

# Output kernel matrix
print(kernel_matrix)
