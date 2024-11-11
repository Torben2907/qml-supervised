import logging
from typing import List
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector


class QuantumFeatureMap(QuantumCircuit):
    def __init__(
        self,
        *,
        feature_dimension: int,
        reps: int,
        num_qubits: int,
        gates: List[str],
        entanglement: str | List[List[int]] = "linear",
        data_scaling: bool = False,
        alpha: float | None = None,
        repeat: bool = True,
        name: str = "Quantum Feature Map",
    ) -> None:
        assert feature_dimension >= 1, "no data sample given."
        assert reps >= 1, "depth of feature map must be larger than 0!"

        self.feature_dimension = feature_dimension
        self.reps = reps
        self._repeat = repeat
        self._gates = gates
        super().__init__(num_qubits, name=name)

        self.data_scaling = data_scaling
        self.alpha = Parameter("⍺")

        if isinstance(entanglement, str):
            self._generate_entanglement(entanglement)
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

        for _ in range(self.reps):
            for g in self._gates:
                if g.isupper():
                    input_encoding = self.build_circuit(
                        g.lower(), self.encoding_params, input_encoding
                    )
                    if self._repeat and input_encoding == self.feature_dimension:
                        input_encoding = 0
                if g.islower():
                    input_train = self.build_circuit(
                        g.lower(), self.training_params, input_train
                    )
            if not self._repeat:
                if input_encoding == self.feature_dimension:
                    input_encoding = 0

        if not self.data_scaling:
            if alpha is not None:
                try:
                    self.assign_parameters({self.alpha: alpha}, inplace=True)
                except ValueError:
                    pass

        return

    @property
    def gates(self) -> List[str]:
        return self._gates

    def _generate_entanglement(self, entanglement: str) -> None:
        self.entanglement = []
        if entanglement == "linear":
            for i in range(self.num_qubits - 1):
                self.entanglement.append([i, i + 1])
        else:
            raise ValueError("unknown method of creating entanglement!")

    def initalize_params(self, gates: List[str]) -> None:
        rotation_gates = ["rx", "ry", "rz"]
        control_gates = ["crx", "cry", "crz", "rxx", "ryy", "rzz", "rzx"]

        num_params = 0
        for g in gates:
            if g.islower():
                if g in rotation_gates:
                    num_params += self.num_qubits
                elif g in control_gates:
                    num_params += len(self.entanglement)

        num_params *= self.reps

        if self.data_scaling:
            num_params += 1

        self.training_params = ParameterVector("θ", num_params)

        if self.data_scaling:
            self.alpha = self.training_params[0]

        t_list = [
            s.isupper() and (s.lower() in rotation_gates + control_gates) for s in gates
        ]

        if not any(t_list):
            logging.warning(msg="Encoding is not specified")

        params = ParameterVector("x", self.feature_dimension)
        self.encoding_params = [self.alpha * p for p in params]
        return

    def build_circuit(self, gate: str, params, start_pos: int = 0):

        def check(pos):
            if pos >= num_params:
                raise CircuitBuilt

        num_params = len(params)
        pos = start_pos

        try:
            _gate = getattr(self, gate)
            if gate in ["crx", "cry", "crz", "rxx", "ryy", "rzz", "rzx", "rzx"]:
                for pair in self.entanglement:
                    if not self.repeat:
                        check(pos)
                    _gate(params[pos % num_params], pair[0], pair[1])
                    pos += 1
            elif gate in ["rx", "ry", "rz"]:
                for i in range(self.num_qubits):
                    if not self.repeat:
                        check(pos)
                    _gate(params[pos % num_params], self.qubits[i])
                    pos += 1
            elif gate in ["x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "t", "tdg"]:
                for i in range(self.num_qubits):
                    _gate(self.qubits[i])
            elif gate in ["cx", "cy", "cz", "ch", "swap", "iswap"]:
                for pair in self.entanglement:
                    _gate(pair[0], pair[1])
        except CircuitBuilt:
            pass

        self.barrier()
        return pos


class CircuitBuilt(Exception):
    pass
