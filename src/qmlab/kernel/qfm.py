from typing import Optional, Sequence
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector


class QuantumFeatureMap(QuantumCircuit):
    def __init__(
        self,
        *,
        num_features: int,
        num_qubits: int,
        num_layers: int,
        gates: Sequence[str],
        entanglement: list | str = "linear",
        data_scaling: bool = False,
        alpha: Optional[float] = None,
        name: str = "Quantum Feature Map",
        repeat: bool = True,
    ) -> None:
        assert num_features >= 1, "no data sample given."
        assert num_layers >= 1, "depth of feature map must be larger than 0!"

        self.num_features = num_features
        self.num_layers = num_layers
        self._repeat = repeat
        super().__init__(num_qubits, name=name)

        self.data_scaling = data_scaling
        self.alpha = Parameter("⍺")

        if isinstance(entanglement, str):
            self.generate_fm(entanglement)
        elif isinstance(entanglement, list):
            self.entanglement = entanglement
        else:
            raise ValueError("entanglement must be of type `list` or `str`.")

        self.initalize_params(gates)

        # setup the circuit
        input_encoding = 0
        if self.data_scaling:
            input_train = 1
        else:
            input_train = 0

        for _ in range(self.num_layers):
            for g in gates:
                if g.isupper():
                    input_encoding = self.build_circuit(
                        g.lower(), self.encoded_params, input_encoding
                    )
                    if self._repeat and input_encoding == self.num_features:
                        input_encoding = 0
                if g.islower():
                    input_train = self.build_circuit(
                        g.lower(), self.train_params, input_train
                    )
            if not self._repeat:
                if input_encoding == self.num_features:
                    input_encoding = 0

        if not self.data_scaling:
            if alpha is not None:
                try:
                    self.assign_parameters({self.alpha: alpha}, inplace=True)
                except:
                    pass
                self.alpha = alpha
        return

    def generate_fm(self, entanglement: str) -> None:
        self.entanglement = []
        if entanglement == "linear":
            for i in range(self.num_qubits - 1):
                self.entanglement.append([i, i + 1])
        else:
            raise ValueError("unknown type for creating entanglement!")

    def initalize_params(self, gates: Sequence[str]) -> None:
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

        if self.data_scaling:
            num_params += 1
            self.alpha = self.train_params[0]

        t_list = [
            g.isupper() and (g.lower() in rotation_gates + controlled_gates)
            for g in gates
        ]

        params = ParameterVector("x", self.num_features)
        self.encoded_params = [self.alpha * p for p in params]
        return

    def build_circuit(self, gate: str, params, j0: int = 0):
        class AllDone(Exception):
            pass

        def check(j):
            if j >= num_params:
                raise AllDone

        num_params = len(params)
        j = j0

        try:
            _gate = getattr(self, gate)
            if gate in ["crx", "cry", "crz", "rxx", "ryy", "rzz", "rzx", "rzx"]:
                for pair in self.entanglement:
                    if not self.repeat:
                        check(j)
                    _gate(params[j % num_params], pair[0], pair[1])
                    j += 1
            elif gate in ["rx", "ry", "rz"]:
                for i in range(self.num_qubits):
                    if not self.repeat:
                        check(j)
                    _gate(params[j % num_params], self.qubits[i])
                    j += 1
            elif gate in ["x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "t", "tdg"]:
                for i in range(self.num_qubits):
                    _gate(self.qubits[i])
            elif gate in ["cx", "cy", "cz", "ch", "swap", "iswap"]:
                for pair in self.entanglement:
                    _gate(pair[0], pair[1])
        except AllDone:
            pass

        self.barrier()
        return j
