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
        self.rotation_params = 2
        self.control_params = 4
        return super().setUp()

    def test_init(self):
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
        assert isinstance(qfm.alpha, Parameter)
        assert isinstance(qfm.training_params, ParameterVector)
        for item in qfm.encoding_params:
            assert isinstance(item, ParameterExpression)

        assert qfm.feature_dimension == self.num_features
        assert qfm.num_qubits == self.num_qubits
        assert qfm.reps == self.num_layers
        assert qfm._repeat == self.repeat
        assert qfm.data_scaling == self.data_scaling

        assert len(qfm.encoding_params) == 2
        assert len(qfm.training_params) == 4

        np.testing.assert_allclose(qfm.entanglement, [[0, 1]])

    def test_initialize_params_1(self):
        qfm = QuantumFeatureMap(feature_dimension=1, num_qubits=1, reps=1, gates=["RX"])
        assert len(qfm.encoding_params) == 1

    def test_initialize_params_2(self):
        qfm = QuantumFeatureMap(
            feature_dimension=1, num_qubits=2, reps=2, gates=["RX", "ry", "crz"]
        )
        assert len(qfm.training_params) == (2 + 4)

    def test_initialize_params_3(self):
        qfm = QuantumFeatureMap(
            feature_dimension=2,
            num_qubits=2,
            reps=2,
            gates=["RX", "ry", "crz", "rx", "rzz"],
        )
        assert len(qfm.encoding_params) == 2
        assert len(qfm.training_params) == (2 + 4 + 2 + 4)
