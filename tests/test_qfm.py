import pytest
import numpy as np
from qmlab.kernel import QuantumFeatureMap
from qmlab_testcase import QMLabTest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector, ParameterExpression


class TestQuantumFeatureMap(QMLabTest):
    def setUp(self):
        self.num_features = 2
        self.num_qubits = self.num_features
        self.num_layers = 2
        self.repeat = True
        self.data_scaling = False
        return super().setUp()

    def test_initialize(self):
        qfm = QuantumFeatureMap(
            num_features=self.num_features,
            num_qubits=self.num_qubits,
            num_layers=self.num_layers,
            gates=["ry", "cz", "RX"],
            entanglement="linear",
            repeat=self.repeat,
            data_scaling=self.data_scaling,
        )

        assert qfm.num_features == self.num_features
        assert qfm.num_qubits == self.num_qubits
        assert qfm.num_layers == self.num_layers
        assert qfm._repeat == self.repeat
        assert qfm.data_scaling == self.data_scaling
        assert isinstance(qfm.alpha, Parameter)
        np.testing.assert_allclose(qfm.entanglement, [[0, 1]])
        for item in qfm.encoded_params:
            assert isinstance(item, ParameterExpression)
