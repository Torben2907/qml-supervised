import numpy as np
from numpy.typing import NDArray
import pennylane as qml


class Wrapper:
    def __init__(self, num_qubits: int = 3):
        self.num_qubits = num_qubits
        self.embedding = qml.AngleEmbedding

    def build_circuit(self):
        dev = qml.device("default.qubit", wires=self.num_qubits)

        @qml.qnode(dev)
        def circuit(x: NDArray):
            self.embedding(x, wires=range(self.num_qubits), rotation="Z")
            return qml.state()

        return circuit

    def evaluate(self, x):
        self.circuit = self.build_circuit()
        return self.circuit(x)


w = Wrapper()
print(w.evaluate([0, np.pi / 2, np.pi]))
