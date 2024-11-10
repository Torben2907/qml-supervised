import numpy as np
from qmlab.qfm import QuantumFeatureMap
from qmlab_testcase import QMLabTest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector, ParameterExpression


class TestQuantumFeatureMap(QMLabTest):
    def setUp(self):
        self.num_features = 2
        self.num_qubits = self.num_features
        self.num_layers = 2
        self.alpha = 1.0
        self.repeat = True
        self.data_scaling = False
        return super().setUp()

    def test_initialize(self):
        qfm = QuantumFeatureMap(
            feature_dimension=self.num_features,
            num_qubits=self.num_qubits,
            reps=self.num_layers,
            gates=["ry", "cz", "RX"],
            entanglement="linear",
            repeat=self.repeat,
            data_scaling=self.data_scaling,
        )

        assert isinstance(qfm, QuantumCircuit)
        assert qfm.feature_dimension == self.num_features
        assert qfm.num_qubits == self.num_qubits
        assert qfm.num_layers == self.num_layers
        assert qfm._repeat == self.repeat
        assert qfm.data_scaling == self.data_scaling
        assert isinstance(qfm.alpha, Parameter)
        np.testing.assert_allclose(qfm.entanglement, [[0, 1]])
        for item in qfm.encoding_params:
            assert isinstance(item, ParameterExpression)
        assert len(qfm.encoding_params) == 2
        assert isinstance(qfm.training_params, ParameterVector)
        assert len(qfm.training_params) == 4
