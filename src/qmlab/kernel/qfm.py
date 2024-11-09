from typing import Sequence
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector


class QuantumFeatureMap(QuantumCircuit):
    def __init__(
        self,
        num_features: int,
        num_qubits: int,
        num_layers: int,
        gates: list[str],
        entanglement: list | str,
        scale: float,
        alpha: float,
        name: str = "Quantum Feature Map",
        repeat: bool = True,
    ) -> None:
        assert num_features >= 1, "no data sample given."
        assert num_layers >= 1, "depth of feature map must be larger than 0!"

        self.num_features = num_features
        self.repeat = repeat

        super().__init__(num_qubits, name=name)

        self.scale = scale
        self.alpha = Parameter("⍺")

        if isinstance(entanglement, str):
            self.generate_fm(entanglement)
        elif isinstance(entanglement, list):
            self.entanglement = entanglement
        else:
            raise ValueError("entanglement must be of type list or str.")

        self.initalize_params(gates)

        return

    def generate_fm(self, entanglement: str) -> None:
        self.entanglement = []
        if entanglement == "linear":
            for i in range(self.num_qubits - 1):
                self.entanglement.append([i, i + 1])
        else:
            raise ValueError("unknown type for creating entanglement!")

    def initalize_params(self, gates: Sequence[str]):
        rotation_gates = ["rx", "ry", "rz"]
        controlled_gates = ["crx", "cry", "crz", "rxx", "ryy", "rzz", "rzx"]

        num_params = 0
        for g in gates:
            if g.islower():
                if g in rotation_gates:
                    num_params += 1 * self.num_qubits
                elif g in controlled_gates:
                    num_params += 2 * len(self.entanglement)

        num_params *= self.num_layers
        self.train_params = ParameterVector("θ", num_params)

        if self.scale:
            num_params += 1
            self.alpha = self.train_params[0]

    def build_circuit(
        self,
    ):
        pass
