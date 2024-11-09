from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def IQPEmbedding(num_features: int, num_qubits: int, num_layers: int):
    feature_map = QuantumCircuit(num_qubits, name="IQPEmbedding")

    x = ParameterVector("x", length=num_features)
    theta = ParameterVector("θ")
    feature_map.theta = theta


class QuantumFeatureMap(QuantumCircuit):
    def __init__(
        self,
        num_features: int,
        num_qubits: int,
        num_layers: int,
        gates: list[str],
        entanglement: list[str] | str,
        scale: float,
        alpha: float,
        name: str,
    ):
        assert num_features >= 1, "no data sample given."
        self.num_features = num_features

    def initalize_params(self):
        pass

    def build_circuit(
        self,
    ):
        pass
