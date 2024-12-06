import numpy as np
import pennylane as qml

dev = qml.device("default.qubit", wires=2)


@qml.qnode(dev)
def bell_circuit1():
    qml.H(0)
    qml.CNOT([0, 1])
    return qml.state()


@qml.qnode(dev)
def bell_circuit2():
    qml.H(1)
    qml.CNOT([1, 0])
    return qml.state()


print(np.allclose(bell_circuit1(), bell_circuit2()))
