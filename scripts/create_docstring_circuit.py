"""used this to create the quantum circuit 
in the FidelityQuantumCircuit
"""

from qiskit import QuantumCircuit
from qiskit.quantum_info import random_unitary

num_qubits = 3
random_u = random_unitary(2**num_qubits).data
circuit = QuantumCircuit(num_qubits)
circuit.unitary(random_u, range(num_qubits), label="U")
circuit.unitary(random_u.conj().T, range(num_qubits), label="U†")
print(circuit.draw(output="text", fold=40, scale=1.5))
