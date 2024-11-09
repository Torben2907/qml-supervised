from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def IQPEmbedding(num_features: int, num_qubits: int, num_layers: int):
    feature_map = QuantumCircuit(num_qubits, name="IQPEmbedding")

    x = ParameterVector("x", length=num_features)
    theta = ParameterVector("θ")
    feature_map.theta = theta
