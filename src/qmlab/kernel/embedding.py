from qiskit import QuantumCircuit


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
