from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector

# Define parameters
theta = Parameter("θ")

# Create a parameterized quantum circuit
qc = QuantumCircuit(1)

init = Statevector(qc)
print(init)

qc.rz(theta, 0)

# Bind parameters
qc_bound = qc.assign_parameters({theta: 0.5})
print(qc.draw())

# Get the state vector
statevector = Statevector(qc_bound)

print(statevector)
