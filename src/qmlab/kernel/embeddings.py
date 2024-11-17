from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def IQPEmbedding(feature_dimension: int, num_qubits: int, reps: int):
    feature_map = QuantumCircuit(num_qubits, name="IQPEmbedding")

    x = ParameterVector("x", length=feature_dimension)
    alpha = ParameterVector("⍺")
    feature_map.alpha = alpha
