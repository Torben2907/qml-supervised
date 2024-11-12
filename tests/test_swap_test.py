import numpy as np

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qmlab.fidelities import SwapTest
from qiskit.primitives import Sampler
from qmlab_testcase import QMLabTest
from qiskit_algorithms.state_fidelities import ComputeUncompute


class TestSwap(QMLabTest):
    def setUp(self):
        super().setUp()
        params = ParameterVector("x", 2)

        rx_rotations = QuantumCircuit(2)
        rx_rotations.rx(params[0], 0)
        rx_rotations.rx(params[1], 1)

        ry_rotations = QuantumCircuit(2)
        ry_rotations.ry(params[0], 0)
        ry_rotations.ry(params[1], 1)

        plus = QuantumCircuit(2)
        plus.h([0, 1])

        zero = QuantumCircuit(2)

        rx_rotation = QuantumCircuit(2)
        rx_rotation.rx(params[0], 0)
        rx_rotation.h(1)

        self._circuit = [rx_rotations, ry_rotations, plus, zero, rx_rotation]
        self._sampler = Sampler()
        self._left_params = np.array(
            [[0, 0], [np.pi / 2, 0], [0, np.pi / 2], [np.pi, np.pi]]
        )
        self._right_params = np.array([[0, 0], [0, 0], [np.pi / 2, 0], [0, 0]])

    def test_param_pair(self):
        fidelity = SwapTest(sampler=self._sampler)
        job = fidelity.run(
            self._circuit[0],
            self._circuit[1],
            self._left_params[0],
            self._right_params[0],
        )
        result = job.result()
        np.testing.assert_allclose(result.fidelities, np.array([1.0]))
